"""Training one run: run here by default, or submitted with --dh."""

from __future__ import annotations

import argparse

import torch
from dhub import submit
from dhub.configs import load_platform
from dhub.publish import model_name, publish_checkpoint
from dhub.store import published_build
from digitalhub_runtime_python import handler

from architecture.mae import CrossSensorMAE
from architecture.models import collate
from config.load import load_config
from dataset.patches import patch_sizes, read_patch_layout
from dataset.store import TRAINING_SPLIT, VALIDATION_SPLIT
from dataset.wavelengths import band_wavelengths
from logs.console import logger
from logs.tracker import start_logging
from training.train import train

TRAINING_HANDLER = "scripts.train:run_training"

_MODEL = load_platform().publishes["model"]

log = logger(__name__)


@handler(outputs=[_MODEL])
def run_training(project=None, overrides: list[str] | None = None):
    """Train one run, and publish the best checkpoint it left.

    Args:
        project: The DigitalHub project the model is logged into, or None here.
        overrides: What to compose the config with, as hydra spells them.

    Returns:
        model: The published checkpoint, or where it was written on a run here.
    """
    config = load_config(overrides or [])
    build = published_build(config.dataset)
    sizes = patch_sizes(config.model.instruments, config.dataset.patchsize)
    shapes, strides = read_patch_layout(build, sizes)
    axes = {name: one.axes for name, one in build.read_row_by_instrument().items()}
    wavelengths = band_wavelengths(shapes, axes)
    loaders = build.loaders_by_split(
        sizes,
        shapes,
        wavelengths,
        collate,
        config.dataset.split,
        config.dataset.seed,
        config.dataset.least_classes,
        config.model.elevation,
        max(config.training.patches_per_step // config.training.batch_size, 1),
        config.dataset.overlap,
        config.training.batch_size,
        config.training.workers,
    )
    training, validation = loaders[TRAINING_SPLIT], loaders[VALIDATION_SPLIT]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("training on %s", device)
    model = CrossSensorMAE(shapes, axes, strides, config.model).to(device)
    run = start_logging(
        config,
        {
            "device": str(device),
            "parameters": sum(one.numel() for one in model.parameters()),
            "shapes": shapes,
            "strides": strides,
            "wavelengths_nm": wavelengths,
            "features": {
                "training": len(training.dataset),
                "validation": len(validation.dataset),
            },
        },
    )
    best = train(model, training, validation, config, device, run)
    run.finish()
    if project is None:
        return best
    return publish_checkpoint(project, best, model_name(config.model))


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
    # The platform calls the handler, a run here the function under it.
    run_training.__wrapped__(overrides=arguments.overrides)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
