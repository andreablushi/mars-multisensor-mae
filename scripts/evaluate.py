"""Evaluating one published model: run here by default, or submitted with --dh."""

from __future__ import annotations

import argparse
from functools import partial

import torch
from dhub import submit
from dhub.publish import model_name
from dhub.store import published_build, published_checkpoint
from digitalhub_runtime_python import handler

from architecture.mae import CrossSensorMAE
from architecture.models import collate
from config.load import load_config
from config.paths import REPO_ROOT
from dataset.patches import patch_sizes, read_patch_layout
from dataset.wavelengths import band_wavelengths
from evaluation.evaluate import evaluate_latent_space, evaluate_reconstruction
from evaluation.report import report_latent_space, report_reconstruction
from logs.console import logger
from logs.tracker import start_logging
from training.checkpoint import load_checkpoint

EVALUATION_HANDLER = "scripts.evaluate:run_evaluation"

log = logger(__name__)


@handler()
def run_evaluation(project=None, overrides: list[str] | None = None) -> None:
    """Measure what one published model's latent space made of the feature classes.

    Args:
        project: The DigitalHub project the model was published in, unused here.
        overrides: What to compose the config with, as hydra spells them.
    """
    config = load_config(overrides or [])
    build = published_build(config.dataset)
    sizes = patch_sizes(config.model.instruments, config.dataset.patchsize)
    shapes, strides = read_patch_layout(build, sizes)
    axes = {name: one.axes for name, one in build.read_row_by_instrument().items()}
    wavelengths = band_wavelengths(shapes, axes)
    read = (
        sizes,
        shapes,
        wavelengths,
        partial(collate, cell_m=config.model.cell_m),
        config.dataset.split,
        config.dataset.seed,
        config.dataset.least_classes,
        config.model.elevation,
        max(config.training.patches_per_step // config.training.batch_size, 1),
        config.dataset.overlap,
        config.training.batch_size,
        config.training.workers,
    )
    # The latents are read over every patch, the reconstruction over one draw of them.
    loader = build.loaders_by_split(*read)[config.evaluation.split]
    ceiling = config.training.patches_per_step
    whole = build.loaders_by_split(*read, ceiling)[config.evaluation.split]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossSensorMAE(shapes, axes, strides, config.model).to(device)
    name = config.evaluation.model or model_name(config.model)
    held = REPO_ROOT / config.training.checkpoints / f"{name}.pt"
    epochs = load_checkpoint(published_checkpoint(name, held), model) + 1
    log.info("evaluating %s, trained for %d epochs, on %s", name, epochs, device)
    run = start_logging(
        config,
        {
            "device": str(device),
            "model": name,
            "epochs_trained": epochs,
            "features": len(loader.dataset),
            "patch_ceiling": ceiling,
        },
    )
    report_latent_space(run, evaluate_latent_space(model, whole, config, device))
    report_reconstruction(
        run,
        evaluate_reconstruction(
            model,
            loader,
            config.training.mask_ratio,
            config.dataset.seed,
            device,
        ),
    )
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
            "evaluation", EVALUATION_HANDLER, arguments.ref, arguments.overrides
        )
    # The platform calls the handler, a run here the function under it.
    run_evaluation.__wrapped__(overrides=arguments.overrides)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
