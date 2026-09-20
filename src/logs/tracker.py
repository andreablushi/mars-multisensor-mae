"""Tracking a run on Weights & Biases, every metric named in one place."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime

import wandb
from dotenv import load_dotenv
from torch import Tensor
from wandb.sdk.wandb_run import Run

from config.paths import REPO_ROOT
from config.schema import Config


def start_logging(config: Config, stage: str, facts: Mapping[str, object]) -> Run:
    """Return the tracked run every metric of one stage is logged to.

    Args:
        config: What the run reads, trains and how, which the run is recorded with.
        stage: What the stage is called, which tells a training from an evaluation.
        facts: What only the assembled run knows, recorded beside the config.

    Returns:
        run: The run, named for the config's own run, the stage and its start, its
            place read from the .env.
    """
    load_dotenv(REPO_ROOT / ".env")
    run = wandb.init(
        config=asdict(config),
        name=f"{config.run}-{stage}-{datetime.now():%Y%m%d-%H%M%S}",
    )
    run.config.update(dict(facts))
    return run


def log_step(run: Run, step: int, terms: Mapping[str, Tensor]) -> None:
    """Log one training step.

    Args:
        run: The tracked run.
        step: Which step of the whole run it is.
        terms: Every loss term, keyed as the loss names them.
    """
    run.log(
        {f"train_{name}": value.item() for name, value in terms.items()},
        step=step,
    )


def log_validation(run: Run, step: int, metrics: Mapping[str, float]) -> None:
    """Log one validation.

    Args:
        run: The tracked run.
        step: Which step of the whole run it was measured after.
        metrics: Every loss term over the validation split, keyed as they name them.
    """
    run.log({f"validation_{name}": value for name, value in metrics.items()}, step=step)


def log_summary(run: Run, summary: Mapping[str, object]) -> None:
    """Write what the whole run came to.

    Args:
        run: The tracked run.
        summary: What it came to, keyed as it is recorded.
    """
    run.summary.update(dict(summary))


def log_latent_space(
    run: Run,
    metrics: Mapping[str, float],
    classes: Sequence[str],
    distances: Sequence[Sequence[float]],
) -> None:
    """Log what a model's latent space came to over the labelled tiles.

    Args:
        run: The tracked run.
        metrics: Every number measured, keyed as it is logged.
        classes: The classes, in the order the distances hold them.
        distances: The mean distance between the tiles of two classes. (C, C)
    """
    table = wandb.Table(columns=["class", *classes])
    for name, row in zip(classes, distances, strict=True):
        table.add_data(name, *(float(one) for one in row))
    run.log(dict(metrics) | {"class_distances": table})
