"""Tracking a run on Weights & Biases."""

from __future__ import annotations

from dataclasses import asdict

import wandb
from dotenv import load_dotenv
from wandb.sdk.wandb_run import Run

from config.paths import REPO_ROOT
from config.schema import Config


def start_run(config: Config) -> Run:
    """Return the tracked run every metric of one training is logged to.

    Args:
        config: What the run reads, trains and how, which the run is recorded
            with.

    Returns:
        run: The run, to log to and to finish. Which entity and project it
            lands under, and the key it presents, are read from the
            environment, or from the `.env` beside the repository on a run
            here: WANDB_ENTITY, WANDB_PROJECT and WANDB_API_KEY.
    """
    load_dotenv(REPO_ROOT / ".env")
    return wandb.init(config=asdict(config))
