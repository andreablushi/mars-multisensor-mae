"""Training one run: run here by default, or submitted with --dh."""

from __future__ import annotations

import argparse
from functools import partial

import torch
from dhub import submit
from dhub.configs import load_platform, stage_workers
from dhub.publish import publish_checkpoint, published_name
from dhub.store import published_build
from digitalhub_runtime_python import handler
from evaluate import evaluate_checkpoint

from architecture.mae import CrossSensorMAE
from architecture.models import collate
from config.load import load_config
from dataset.loader import TRAINING_SPLIT, VALIDATION_SPLIT, loaders_by_split
from dataset.patches import patch_sizes, read_patch_layout
from logs.console import rich_logger
from logs.tracker import start_logging
from training.train import train

TRAINING_STAGE = "training"
TRAINING_HANDLER = "scripts.train:run_training"

_MODEL = load_platform().publishes["model"]

log = rich_logger(__name__)


@handler(outputs=[_MODEL])
def run_training(
    project=None, overrides: list[str] | None = None, evaluate: bool = False
):
    """Train one run, publish the best checkpoint it left, and evaluate it if asked.

    Args:
        project: The DigitalHub project the model is logged into, or None here.
        overrides: What to compose the config with, as hydra spells them.
        evaluate: Whether to evaluate the best checkpoint once the training ends.

    Returns:
        model: The published checkpoint, or where it was written on a run here.
    """
    config = load_config(overrides or [])
    build = published_build(config.dataset.build, config.dataset.root)
    sizes = patch_sizes(config.model.instruments, config.dataset.patchsize)
    shapes, strides = read_patch_layout(build, sizes)
    axes = build.read_axes_by_instrument()
    loaders = loaders_by_split(
        build,
        sizes,
        shapes,
        partial(collate, cell_m=config.model.cell_m),
        config.dataset.split,
        config.dataset.seed,
        config.model.delay,
        config.training.batch_size,
        stage_workers(TRAINING_STAGE),
    )
    training, validation = loaders[TRAINING_SPLIT], loaders[VALIDATION_SPLIT]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("training on %s", device)
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
    run = start_logging(
        config,
        TRAINING_STAGE,
        {
            "device": str(device),
            "parameters": sum(one.numel() for one in model.parameters()),
            "shapes": shapes,
            "strides": strides,
            "tiles": {
                "training": len(training.dataset),
                "validation": len(validation.dataset),
            },
        },
    )
    best = train(
        model,
        training,
        validation,
        config.training.max_steps,
        config.training.learning_rate,
        config.training.weight_decay,
        config.training.warmup_steps,
        config.training.validate_every,
        config.training.patience,
        config.training.mask_ratio,
        config.training.checkpoints,
        config.dataset.seed,
        device,
        run,
    )
    run.finish()
    published = best
    if project is not None:
        name = published_name("model", config.run_name)
        published = publish_checkpoint(project, best, name)
    if evaluate:
        evaluate_checkpoint(config, best, project)
    return published


def main() -> int:
    """Run the training where it was asked for.

    Returns:
        code: A process exit code, non zero when the image did not build.
    """
    parsed = argparse.ArgumentParser(description=__doc__)
    parsed.add_argument(
        "--dh", action="store_true", help="submit to DigitalHub instead of running here"
    )
    parsed.add_argument("--ref", default="main", help="branch, tag, or commit to run")
    parsed.add_argument(
        "--evaluate", action="store_true", help="evaluate the model once trained"
    )
    parsed.add_argument(
        "overrides",
        nargs="*",
        help="what to compose the config with, as hydra spells them",
    )
    arguments = parsed.parse_args()
    if arguments.dh:
        return submit.submitted(
            TRAINING_STAGE,
            TRAINING_HANDLER,
            arguments.ref,
            {"overrides": arguments.overrides, "evaluate": arguments.evaluate},
        )
    # The platform calls the handler, a run here the function under it.
    run_training.__wrapped__(overrides=arguments.overrides, evaluate=arguments.evaluate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
