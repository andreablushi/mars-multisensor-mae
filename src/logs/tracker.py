"""Tracking a run on Weights & Biases."""

from __future__ import annotations

from dataclasses import asdict

import wandb
from wandb.sdk.wandb_run import Run

from config.schema import Config


def start_run(config: Config) -> Run:
    """Return the tracked run every metric of one training is logged to.

    Args:
        config: What the run reads, trains and how, which the run is recorded
            with, under the project the training names.

    Returns:
        run: The run, to log to and to finish.
    """
    return wandb.init(
        project=config.training.project,
        entity=config.training.entity,
        config=asdict(config),
    )
