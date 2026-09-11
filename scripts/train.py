"""Training one run: run here by default, or submitted with --dh."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from dh import submit
from dh.configs import load_platform
from dh.publish import publish_checkpoint
from dh.store import published_build
from digitalhub_runtime_python import handler

from architecture.mae import CrossSensorMAE
from architecture.tokens import collate
from config.load import load_config
from config.schema import Config
from dataset.patches import patch_lengths, patch_sizes
from dataset.store import TRAINING_SPLIT, VALIDATION_SPLIT
from logs.console import logger
from logs.tracker import start_run
from training.train import train

TRAINING_HANDLER = "scripts.train:run_training"

_MODEL = load_platform().publishes["model"]

log = logger(__name__)


def train_model(config: Config) -> Path:
    """Return the best checkpoint of one run, trained as the config describes it.

    Args:
        config: What the run reads, trains and how.

    Returns:
        best: The checkpoint with the lowest validation loss.
    """
    build = published_build(config.dataset)
    sizes = patch_sizes(config.model.instruments, config.dataset.patchsize)
    rows = build.read_row_by_instrument()
    shapes = {
        name: patch_lengths(rows[name].shape, rows[name].axes, size)
        for name, size in sizes.items()
    }
    ground = build.read_ground_sample_by_instrument()
    strides = {name: size * ground[name] for name, size in sizes.items()}
    loaders = build.loaders_by_split(
        config.dataset.split,
        config.dataset.seed,
        sizes,
        shapes,
        config.model.elevation,
        config.model.patches,
        config.training.batch_size,
        config.training.workers,
        collate,
    )
    training, validation = loaders[TRAINING_SPLIT], loaders[VALIDATION_SPLIT]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("training on %s", device)
    model = CrossSensorMAE(shapes, strides, config.model).to(device)
    run = start_run(
        config,
        {
            "device": str(device),
            "parameters": sum(one.numel() for one in model.parameters()),
            "shapes": shapes,
            "strides": strides,
            "features": {
                "training": len(training.dataset),
                "validation": len(validation.dataset),
            },
        },
    )
    best = train(model, training, validation, config, device, run)
    run.finish()
    return best


@handler(outputs=[_MODEL])
def run_training(project, overrides: list[str] | None = None):
    """Train one run on DigitalHub and publish the best checkpoint it left.

    Args:
        project: The DigitalHub project the model is logged into.
        overrides: What to compose the config with, as hydra spells them.

    Returns:
        model: The published checkpoint.
    """
    config = load_config(overrides or [])
    best = train_model(config)
    return publish_checkpoint(project, best, f"{_MODEL}-{config.model.name}")


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
        "overrides",
        nargs="*",
        help="what to compose the config with, as hydra spells them",
    )
    arguments = parsed.parse_args()
    if arguments.dh:
        return submit.submitted(
            "training", TRAINING_HANDLER, arguments.ref, arguments.overrides
        )
    train_model(load_config(arguments.overrides))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
