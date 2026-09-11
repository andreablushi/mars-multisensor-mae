"""Evaluating one published model: run here by default, or submitted with --dh."""

from __future__ import annotations

import argparse

import torch
from dh import submit
from dh.publish import model_name
from dh.store import published_build, published_checkpoint
from digitalhub_runtime_python import handler

from architecture.mae import CrossSensorMAE
from architecture.tokens import collate
from config.load import load_config
from config.paths import REPO_ROOT
from config.schema import Config
from dataset.patches import patch_sizes
from evaluation.evaluate import evaluate_latent_space
from evaluation.report import report_evaluation
from logs.console import logger
from logs.tracker import start_run
from training.checkpoint import load_checkpoint

EVALUATION_HANDLER = "scripts.evaluate:run_evaluation"

log = logger(__name__)


def evaluate_model(config: Config) -> None:
    """Measure what one published model's latent space made of the feature classes.

    Args:
        config: What the run reads, what it built, which model is read and how
            it is measured.
    """
    build = published_build(config.dataset)
    sizes = patch_sizes(config.model.instruments, config.dataset.patchsize)
    shapes, strides = build.read_patch_layout(sizes)
    loader = build.loaders_by_split(config, sizes, shapes, collate)[
        config.evaluation.split
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossSensorMAE(shapes, strides, config.model).to(device)
    name = config.evaluation.model or model_name(config.model)
    held = REPO_ROOT / config.training.checkpoints / f"{name}.pt"
    epochs = load_checkpoint(published_checkpoint(name, held), model) + 1
    log.info("evaluating %s, trained for %d epochs, on %s", name, epochs, device)
    run = start_run(
        config,
        {
            "device": str(device),
            "model": name,
            "epochs_trained": epochs,
            "features": len(loader.dataset),
        },
    )
    report_evaluation(run, evaluate_latent_space(model, loader, config, device))
    run.finish()


@handler()
def run_evaluation(project, overrides: list[str] | None = None) -> None:
    """Evaluate one published model on DigitalHub.

    Args:
        project: The DigitalHub project the model was published in.
        overrides: What to compose the config with, as hydra spells them.
    """
    evaluate_model(load_config(overrides or []))


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
    evaluate_model(load_config(arguments.overrides))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
