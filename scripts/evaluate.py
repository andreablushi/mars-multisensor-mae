"""Evaluating one published model: run here by default, or submitted with --dh."""

from __future__ import annotations

import argparse
from functools import partial

import torch
from dhub import submit
from dhub.configs import stage_workers
from dhub.publish import model_name
from dhub.store import published_build, published_checkpoint
from digitalhub_runtime_python import handler

from architecture.mae import CrossSensorMAE
from architecture.models import collate
from config.load import load_config
from config.paths import REPO_ROOT
from dataset.patches import patch_sizes, read_patch_layout
from evaluation.evaluate import evaluate_latent_space
from logs.console import rich_logger
from logs.tracker import start_logging
from training.checkpoint import load_checkpoint

EVALUATION_STAGE = "evaluation"
EVALUATION_HANDLER = "scripts.evaluate:run_evaluation"

log = rich_logger(__name__)


@handler()
def run_evaluation(project=None, overrides: list[str] | None = None) -> None:
    """Measure what one published model's latent space made of a split of tiles.

    Args:
        project: The DigitalHub project the model was published in, unused here.
        overrides: What to compose the config with, as hydra spells them.
    """
    config = load_config(overrides or [])
    build = published_build(config.dataset)
    sizes = patch_sizes(config.model.instruments, config.dataset.patchsize)
    shapes, strides = read_patch_layout(build, sizes)
    axes = {name: one.axes for name, one in build.read_row_by_instrument().items()}
    loader = build.loaders_by_split(
        sizes,
        shapes,
        partial(collate, cell_m=config.model.cell_m),
        config.dataset.split,
        config.dataset.seed,
        config.model.delay,
        config.training.batch_size,
        stage_workers(EVALUATION_STAGE),
    )[config.evaluation.split]
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
    name = model_name(config.run)
    held = REPO_ROOT / config.training.checkpoints / f"{name}.pt"
    steps = load_checkpoint(published_checkpoint(name, held), model)
    log.info("evaluating %s, trained for %d steps, on %s", name, steps, device)
    run = start_logging(
        config,
        EVALUATION_STAGE,
        {
            "device": str(device),
            "checkpoint": name,
            "steps_trained": steps,
            "tiles": len(loader.dataset),
        },
    )
    grids = evaluate_latent_space(model, loader, device)
    run.log({"latent/tiles": len(grids)})
    run.finish()


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
            EVALUATION_STAGE, EVALUATION_HANDLER, arguments.ref, arguments.overrides
        )
    # The platform calls the handler, a run here the function under it.
    run_evaluation.__wrapped__(overrides=arguments.overrides)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
