"""Evaluating one published model: run here by default, or submitted with --dh."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import torch
from dhub import submit
from dhub.configs import stage_workers
from dhub.publish import publish_results, published_name
from dhub.store import published_build, published_checkpoint
from digitalhub_runtime_python import handler

from architecture.mae import CrossSensorMAE
from architecture.models import collate
from config.load import load_config
from config.paths import REPO_ROOT, RESULTS_ROOT
from config.schema import Config
from dataset.loader import read_training_statistics, tile_loader
from dataset.patches import patch_sizes, read_patch_layout
from evaluation.evaluate import evaluate_latent_space, measure_latent_space
from evaluation.results import RESULTS_FILE, write_tile_distances
from evaluation.store import read_label_by_tile
from logs.console import rich_logger
from logs.tracker import log_latent_space, start_logging
from training.checkpoint import load_checkpoint

EVALUATION_STAGE = "evaluation"
EVALUATION_HANDLER = "scripts.evaluate:run_evaluation"

log = rich_logger(__name__)


def evaluate_checkpoint(config: Config, checkpoint: Path, project=None) -> None:
    """Measure one checkpoint's latent space over the labelled tiles, and keep it.

    Args:
        config: What the run reads, the model it builds, and how it is measured.
        checkpoint: The checkpoint to measure, on this machine.
        project: The DigitalHub project the results are published in, or None here.
    """
    # The model is built and normalised as the training build left it, whichever
    # build the tiles it never read come from.
    trained = published_build(config.dataset.build, config.dataset.root)
    sizes = patch_sizes(config.model.instruments, config.dataset.patchsize)
    shapes, strides = read_patch_layout(trained, sizes, config.dataset.pool)
    axes = trained.read_axes_by_instrument()
    statistics = read_training_statistics(
        trained, config.dataset.split, config.dataset.seed
    )
    build = published_build(config.evaluation.build, config.dataset.root)
    classes = read_label_by_tile(build)
    by_tile = build.read_observation_metadata_by_tile()
    loader = tile_loader(
        build,
        {tile: rows for tile, rows in by_tile.items() if tile in classes},
        axes,
        statistics,
        sizes,
        config.dataset.pool,
        shapes,
        partial(collate, cell_m=config.model.cell_m),
        config.model.delay,
        config.training.batch_size,
        stage_workers(EVALUATION_STAGE),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossSensorMAE(
        shapes,
        axes,
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
    steps = load_checkpoint(checkpoint, model)
    log.info("evaluating %s, trained for %d steps, on %s", checkpoint, steps, device)
    run = start_logging(
        config,
        EVALUATION_STAGE,
        {
            "device": str(device),
            "checkpoint": checkpoint.name,
            "steps_trained": steps,
            "tiles": len(loader.dataset),
            "classes": len(set(classes.values())),
        },
    )
    grids = evaluate_latent_space(model, loader, device)
    measured = measure_latent_space(
        grids,
        classes,
        config.evaluation.minimal_chamfer_cell_distance,
        config.dataset.seed,
    )
    log_latent_space(
        run,
        measured.metrics,
        measured.classes,
        measured.distances,
        measured.tiles,
        [classes[tile] for tile in measured.tiles],
        measured.projection,
    )
    run.finish()
    results = RESULTS_ROOT / config.run_name / RESULTS_FILE
    write_tile_distances(results, measured.tiles, classes, measured.tile_distances)
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


def main() -> int:
    """Run the evaluation where it was asked for.

    Returns:
        code: A process exit code, non zero when the image did not build.
    """
    parsed = argparse.ArgumentParser(description=__doc__)
    parsed.add_argument(
        "--dh", action="store_true", help="submit to DigitalHub instead of running here"
    )
    parsed.add_argument("--ref", default="main", help="branch, tag, or commit to run")
    parsed.add_argument(
        "overrides",
        nargs="*",
        help="what to compose the config with, as hydra spells them",
    )
    arguments = parsed.parse_args()
    if arguments.dh:
        return submit.submitted(
            EVALUATION_STAGE,
            EVALUATION_HANDLER,
            arguments.ref,
            {"overrides": arguments.overrides},
        )
    # The platform calls the handler, a run here the function under it.
    run_evaluation.__wrapped__(overrides=arguments.overrides)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
