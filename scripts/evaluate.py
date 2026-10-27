"""Evaluating one logged model: run here by default, or submitted with --dh."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict

import torch
from dhub.store import published_build
from dhub.submit import run_stage, stage_workers
from digitalhub_runtime_python import handler

from architecture.mae import CrossSensorMAE
from configs.load import load_config
from configs.paths import ATTENTION_FILE, RECONSTRUCTION_FILE, results_path
from configs.schema import Config
from dataset.loader import cache_tiles, tile_loader
from dataset.store import DatasetBuild
from evaluation.evaluate import evaluate
from evaluation.reconstruct import reconstruction_and_attention
from evaluation.results import write_tile_arrays, write_tile_vectors
from logs.console import console_logger
from logs.evaluation import BASELINE, EVALUATION, log_evaluation
from logs.tracker import logged_checkpoint, start_logging

EVALUATION_HANDLER = "scripts.evaluate:run_evaluation"
EVALUATION_BATCH_SIZE = 1

log = console_logger(__name__)


def built_model(
    config: Config, build: DatasetBuild
) -> tuple[
    CrossSensorMAE,
    torch.device,
    dict[str, Mapping[str, int]],
    dict[str, tuple[int, ...]],
]:
    """Return the model a build's patch layout settles, on the GPU when there is one.

    Args:
        config: What the run reads and the model it builds.
        build: The build whose instruments settle the patch layout.

    Returns:
        model: The model, on its device.
        device: Where it runs.
        sizes: How far a patch of each sensor runs along each axis it is cut on.
        shapes: The shape of one patch of each instrument as the model reads it.
    """
    sizes = {name: config.dataset.patchsize[name] for name in config.model.instruments}
    shapes = build.read_patch_shapes(sizes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossSensorMAE(
        shapes,
        {name: build.condition_width(name, shape) for name, shape in shapes.items()},
        config.model.film,
        config.model.encoder_dim,
        config.model.encoder_heads,
        config.model.encoder_depth,
        config.model.crossencoder_depth,
        config.model.decoder_dim,
        config.model.decoder_heads,
        config.model.decoder_depth,
        config.dataset.seed,
    ).to(device)
    return model, device, sizes, shapes


@handler()
def run_evaluation(run_name: str, overrides: list[str] | None = None) -> None:
    """Measure one logged model's latent space over the labelled tiles, rebuild some.

    Args:
        run_name: What the run is called, which names the model read and the
            tracked run, a last part of "random" measuring the seed's own weights.
        overrides: What to change in the configs, as dotted keys.
    """
    config = load_config(overrides or [])
    build = published_build(
        config.evaluation.build,
        config.dataset.root,
        config.dataset.clip,
        config.model.clutter,
    )
    model, device, sizes, shapes = built_model(config, build)
    workers = stage_workers(EVALUATION)
    cache_tiles(build, sizes, shapes, workers)
    by_tile = build.read_observation_metadata()
    loader = tile_loader(
        build,
        list(by_tile),
        sizes,
        EVALUATION_BATCH_SIZE,
        workers,
        shuffle=False,
        budget=None,
        scale=1.0,
        seed=None,
    )
    run = start_logging(asdict(config), run_name, EVALUATION, {"device": str(device)})
    if run_name.split("/")[-1] != BASELINE:
        held = torch.load(logged_checkpoint(run, run_name), map_location="cpu")
        model.load_state_dict(held["model"])
        log.info("%s trained for %d steps", run_name, held["step"])
    log.info("evaluating %s on %s", run_name, device)
    vectors = evaluate(model, loader, device)
    classes = build.read_label_by_tile()
    files = [results_path(run_name)]
    write_tile_vectors(files[0], classes, vectors)
    reconstructions, attentions = {}, {}
    for place, tile in config.evaluation.tiles.items():
        reconstructions[place], by_query = reconstruction_and_attention(
            model,
            build,
            by_tile[tile],
            sizes,
            shapes,
            config.training.mask_ratio,
            config.dataset.seed,
            device,
        )
        files.append(results_path(run_name, RECONSTRUCTION_FILE.format(place)))
        write_tile_arrays(files[-1], reconstructions[place])
        for query, attention in by_query.items():
            attentions[place, query] = attention
            files.append(results_path(run_name, ATTENTION_FILE.format(place, query)))
            write_tile_arrays(files[-1], attention)
    log_evaluation(run, run_name, vectors, classes, reconstructions, attentions, files)
    run.finish()


if __name__ == "__main__":
    run_stage(EVALUATION, EVALUATION_HANDLER, run_evaluation, __doc__)
