"""Evaluating one published model: run here by default, or submitted with --dh."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from dhub.store import publish_results, published_build, published_checkpoint
from dhub.submit import run_stage, stage_workers
from digitalhub_runtime_python import handler

from architecture.mae import CrossSensorMAE
from configs.load import load_config
from configs.paths import results_path
from configs.schema import Config
from dataset.loader import tile_loader
from dataset.patches import read_patch_shapes
from dataset.store import DatasetBuild
from evaluation.evaluate import evaluate_latent_space
from evaluation.results import write_tile_vectors
from logs.console import console_logger
from training.checkpoint import load_checkpoint

EVALUATION_STAGE = "evaluation"
EVALUATION_HANDLER = "scripts.evaluate:run_evaluation"
RANDOM = "random"

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
    shapes = read_patch_shapes(build, sizes, config.dataset.pool)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossSensorMAE(
        shapes,
        config.model.encoder_dim,
        config.model.encoder_heads,
        config.model.encoder_depth,
        config.model.crossencoder_depth,
        config.model.decoder_dim,
        config.model.decoder_heads,
        config.model.decoder_depth,
    ).to(device)
    return model, device, sizes, shapes


def evaluate_checkpoint(config: Config, checkpoint: Path | None, project=None) -> None:
    """Measure one checkpoint's latent space over the labelled tiles, and keep it.

    Args:
        config: What the run reads, the model it builds, and how it is measured.
        checkpoint: The checkpoint to measure, on this machine, or None to measure
            the weights the run's seed draws.
        project: The DigitalHub project the results are published in, or None here.
    """
    # The model is built as the training build left it, whichever build the tiles
    # it never read come from.
    trained = published_build(config.dataset.build, config.dataset.root)
    torch.manual_seed(config.dataset.seed)
    model, device, sizes, shapes = built_model(config, trained)
    axes = trained.read_axes_by_instrument()
    build = published_build(config.evaluation.build, config.dataset.root)
    classes = build.read_label_by_tile()
    by_tile = build.read_observation_metadata()
    loader = tile_loader(
        build,
        {tile: rows for tile, rows in by_tile.items() if tile in classes},
        axes,
        sizes,
        config.dataset.pool,
        shapes,
        config.model.delay,
        config.training.memory_batch_size,
        stage_workers(EVALUATION_STAGE),
        shuffle=False,
        budget=None,
        seed=None,
    )
    if checkpoint is not None:
        steps = load_checkpoint(checkpoint, model)
        log.info("%s trained for %d steps", checkpoint, steps)
    log.info("evaluating %s on %s", config.run_name, device)
    results = results_path(config.run_name)
    write_tile_vectors(results, classes, evaluate_latent_space(model, loader, device))
    log.info("results written to %s", results)
    if project is not None:
        publish_results(project, results, config.run_name)


@handler()
def run_evaluation(project=None, overrides: list[str] | None = None) -> None:
    """Measure what one published model's latent space made of the labelled tiles.

    Args:
        project: The DigitalHub project the model was published in, or None here.
        overrides: What to compose the config with, as hydra spells them.
    """
    config = load_config(overrides or [])
    checkpoint = None
    if config.run_name != RANDOM:
        checkpoint = published_checkpoint(config.run_name, config.training.checkpoints)
    evaluate_checkpoint(config, checkpoint, project)


if __name__ == "__main__":
    run_stage(EVALUATION_STAGE, EVALUATION_HANDLER, run_evaluation, __doc__)
