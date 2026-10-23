"""Evaluating one published model: run here by default, or submitted with --dh."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from building.common.layout import Axis
from dhub.store import publish_results, published_build, published_checkpoint
from dhub.submit import run_stage, stage_workers
from digitalhub_runtime_python import handler

from architecture.mae import CrossSensorMAE
from configs.load import load_config
from configs.paths import RECONSTRUCTION_FILE, results_path
from configs.schema import Config
from dataset.loader import tile_loader
from dataset.patches import read_patch_shapes
from dataset.store import DatasetBuild
from evaluation.evaluate import evaluate
from evaluation.reconstruct import reconstruct
from evaluation.results import write_reconstruction, write_tile_vectors
from logs.console import console_logger

EVALUATION_STAGE = "evaluation"
EVALUATION_HANDLER = "scripts.evaluate:run_evaluation"
RANDOM = "random"
RECONSTRUCTED_CLASS = "crater"
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
    shapes = read_patch_shapes(build, sizes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossSensorMAE(
        shapes,
        {name: build.condition_width(name, shape) for name, shape in shapes.items()},
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


def evaluate_checkpoint(
    config: Config, run_name: str, checkpoint: Path | None, project=None
) -> None:
    """Measure one checkpoint's latent space over the labelled tiles, rebuild one.

    Args:
        config: What the run reads, the model it builds, and how it is measured.
        run_name: What the run is called, which names its results.
        checkpoint: The checkpoint to measure, on this machine, or None to measure
            the weights the run's seed draws.
        project: The DigitalHub project the results are published in, or None here.
    """
    # The model is built as the training build left it, whichever build the tiles
    # it never read come from.
    trained = published_build(
        config.dataset.build,
        config.dataset.root,
        config.dataset.clip,
        config.model.clutter_conditioning,
    )
    model, device, sizes, shapes = built_model(config, trained)
    axes = trained.read_axes_by_instrument()
    build = published_build(
        config.evaluation.build,
        config.dataset.root,
        config.dataset.clip,
        config.model.clutter_conditioning,
    )
    classes = build.read_label_by_tile()
    by_tile = build.read_observation_metadata()
    loader = tile_loader(
        build,
        by_tile,
        axes,
        sizes,
        shapes,
        EVALUATION_BATCH_SIZE,
        stage_workers(EVALUATION_STAGE),
        shuffle=False,
        budget=None,
        scale=1.0,
        seed=None,
    )
    if checkpoint is not None:
        held = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(held["model"])
        log.info("%s trained for %d steps", checkpoint, held["step"])
    log.info("evaluating %s on %s", run_name, device)
    results = results_path(run_name)
    write_tile_vectors(results, classes, evaluate(model, loader, device))
    # The tile of the class seen by the most ground observations
    tile = max(
        sorted(one for one in by_tile if classes[one] == RECONSTRUCTED_CLASS),
        key=lambda one: sum(
            len(by_tile[one][name]) for name in sizes if Axis.DELAY not in axes[name]
        ),
    )
    reconstruction = results_path(run_name, RECONSTRUCTION_FILE)
    write_reconstruction(
        reconstruction,
        tile,
        reconstruct(
            model,
            build,
            by_tile[tile],
            axes,
            sizes,
            shapes,
            config.training.mask_ratio,
            config.dataset.seed,
            device,
        ),
    )
    log.info("results written to %s", results.parent)
    if project is not None:
        for path in (results, reconstruction):
            publish_results(project, path, run_name)


@handler()
def run_evaluation(
    run_name: str, project=None, overrides: list[str] | None = None
) -> None:
    """Measure what one published model's latent space made of the labelled tiles.

    Args:
        run_name: What the run is called, which names the model read and its
            results, a last part of "random" measuring the seed's own weights.
        project: The DigitalHub project the model was published in, or None here.
        overrides: What to change in the configs, as dotted keys.
    """
    config = load_config(overrides or [])
    checkpoint = None
    if run_name.split("/")[-1] != RANDOM:
        checkpoint = published_checkpoint(run_name, config.training.checkpoints)
    evaluate_checkpoint(config, run_name, checkpoint, project)


if __name__ == "__main__":
    run_stage(EVALUATION_STAGE, EVALUATION_HANDLER, run_evaluation, __doc__)
