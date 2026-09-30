"""Evaluating one published model: run here by default, or submitted with --dh."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path

import torch
from dhub.configs import stage_workers
from dhub.publish import publish_results, published_name
from dhub.store import published_build, published_checkpoint
from dhub.submit import ran_stage
from digitalhub_runtime_python import handler
from torch.utils.data import DataLoader

from architecture.mae import CrossSensorMAE
from architecture.tokens import token_batch_padding
from configs.load import load_config
from configs.paths import REPO_ROOT, RESULTS_ROOT
from configs.schema import Config
from dataset.models.split import DatasetSplit
from dataset.patches import read_patch_layout
from dataset.store import DatasetBuild
from evaluation.evaluate import evaluate_latent_space
from evaluation.metrics import chamfer_distances
from evaluation.results import RESULTS_FILE, write_tile_distances
from evaluation.store import read_label_by_tile
from logs.console import console_logger
from training.checkpoint import load_checkpoint

EVALUATION_STAGE = "evaluation"
EVALUATION_HANDLER = "scripts.evaluate:run_evaluation"

log = console_logger(__name__)


def built_model(
    config: Config, build: DatasetBuild
) -> tuple[
    CrossSensorMAE,
    torch.device,
    dict[str, Mapping[str, int]],
    dict[str, tuple[int, ...]],
    dict[str, float],
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
        strides: How far apart two neighbouring patch centres of each sensor sit.
    """
    sizes = {name: config.dataset.patchsize[name] for name in config.model.instruments}
    shapes, strides, centres_nm = read_patch_layout(build, sizes, config.dataset.pool)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossSensorMAE(
        shapes,
        build.read_axes_by_instrument(),
        centres_nm,
        strides,
        config.model.encoder_dim,
        config.model.encoder_heads,
        config.model.encoder_depth,
        config.model.crossencoder_depth,
        config.model.decoder_dim,
        config.model.decoder_heads,
        config.model.decoder_depth,
        config.model.cell_m,
    ).to(device)
    return model, device, sizes, shapes, strides


def evaluate_checkpoint(config: Config, checkpoint: Path, project=None) -> None:
    """Measure one checkpoint's latent space over the labelled tiles, and keep it.

    Args:
        config: What the run reads, the model it builds, and how it is measured.
        checkpoint: The checkpoint to measure, on this machine.
        project: The DigitalHub project the results are published in, or None here.
    """
    # The model is built as the training build left it, whichever build the tiles
    # it never read come from.
    trained = published_build(config.dataset.build, config.dataset.root)
    model, device, sizes, shapes, _ = built_model(config, trained)
    axes = trained.read_axes_by_instrument()
    build = published_build(config.evaluation.build, config.dataset.root)
    classes = read_label_by_tile(build)
    by_tile = build.read_observation_metadata_by_tile()
    workers = stage_workers(EVALUATION_STAGE)
    loader = DataLoader(
        DatasetSplit(
            build,
            {tile: rows for tile, rows in by_tile.items() if tile in classes},
            axes,
            sizes,
            config.dataset.pool,
            shapes,
            config.model.delay,
        ),
        batch_size=config.training.batch_size,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=True,
        collate_fn=partial(
            token_batch_padding,
            cell_m=config.model.cell_m,
            delay_rows=config.dataset.patchsize["SHARAD"]["delay"],
            full_grid=True,
        ),
    )
    steps = load_checkpoint(checkpoint, model)
    log.info("evaluating %s, trained for %d steps, on %s", checkpoint, steps, device)
    grids = evaluate_latent_space(
        model,
        loader,
        device,
        config.dataset.patchsize["SHARAD"]["delay"],
        config.evaluation.delay_window,
    )
    tiles = sorted(grids)
    distances = chamfer_distances(
        [grids[tile] for tile in tiles], config.evaluation.minimal_chamfer_cell_distance
    )
    results = RESULTS_ROOT / config.run_name / RESULTS_FILE
    write_tile_distances(results, tiles, classes, distances.double().cpu().numpy())
    log.info("results written to %s", results)
    if project is not None:
        publish_results(project, results, published_name("results", config.run_name))


@handler()
def run_evaluation(project=None, overrides: list[str] | None = None) -> None:
    """Measure what one published model's latent space made of the labelled tiles.

    Args:
        project: The DigitalHub project the model was published in, or None here.
        overrides: What to compose the config with, as hydra spells them.
    """
    config = load_config(overrides or [])
    name = published_name("model", config.run_name)
    held = REPO_ROOT / config.training.checkpoints / f"{name}.pt"
    evaluate_checkpoint(config, published_checkpoint(name, held), project)


if __name__ == "__main__":
    raise SystemExit(
        ran_stage(EVALUATION_STAGE, EVALUATION_HANDLER, run_evaluation, __doc__)
    )
