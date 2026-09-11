"""Tracking a run on Weights & Biases, every metric named in one place."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict

import wandb
from dotenv import load_dotenv
from torch import Tensor
from wandb.sdk.wandb_run import Run

from config.paths import REPO_ROOT
from config.schema import Config


def start_run(config: Config, facts: Mapping[str, object]) -> Run:
    """Return the tracked run every metric of one training is logged to.

    Args:
        config: What the run reads, trains and how, which the run is recorded
            with.
        facts: What only the assembled run knows, recorded beside the config:
            the device, the parameter count, the patch shapes and how many
            features each split holds.

    Returns:
        run: The run, to log to and to finish. Which entity and project it
            lands under, and the key it presents, are read from the
            environment, or from the `.env` beside the repository on a run
            here: WANDB_ENTITY, WANDB_PROJECT and WANDB_API_KEY.
    """
    load_dotenv(REPO_ROOT / ".env")
    run = wandb.init(config=asdict(config))
    run.config.update(dict(facts))
    return run


def log_step(
    run: Run, step: int, terms: Mapping[str, Tensor], measured: Mapping[str, float]
) -> None:
    """Log one training step.

    Args:
        run: The tracked run.
        step: Which step of the whole run it is.
        terms: Every loss term, keyed as the loss names them.
        measured: What was measured around the step, keyed as it is logged.
    """
    logged = {f"train/{name}": float(value) for name, value in terms.items()}
    run.log(logged | dict(measured), step=step)


def log_epoch(
    run: Run,
    step: int,
    epoch: int,
    metrics: Mapping[str, float],
    loads: Sequence[float],
    seconds: float,
) -> None:
    """Log one epoch's validation, and how the epoch that led to it read.

    Args:
        run: The tracked run.
        step: Which step of the whole run the epoch ended on.
        epoch: Which pass over the training split it was.
        metrics: Every loss term and retrieval metric over the validation
            split, keyed as they name them.
        loads: How long each feature of the epoch took to read, in seconds.
        seconds: How long the epoch took, training and validation together.
    """
    logged = {f"validation/{name}": value for name, value in metrics.items()}
    run.log(
        logged
        | {
            "epoch": epoch,
            "time/epoch_seconds": seconds,
            "data/feature_seconds_spread": wandb.Histogram(list(loads)),
        },
        step=step,
    )


def log_summary(run: Run, summary: Mapping[str, object]) -> None:
    """Write what the whole run came to.

    Args:
        run: The tracked run.
        summary: What it came to, keyed as it is recorded.
    """
    run.summary.update(dict(summary))
