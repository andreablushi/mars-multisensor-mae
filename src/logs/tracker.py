"""Tracking a run on Weights & Biases, every metric named in one place."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import wandb
from dotenv import load_dotenv
from torch import Tensor
from wandb.sdk.wandb_run import Run

from config.paths import REPO_ROOT
from config.schema import Config


def run_name() -> str:
    """Return what one run is tracked under, the branch it runs and when it started.

    Returns:
        name: The branch, or the short commit where the head names none, and
            the local date and time the run started, to the second.
    """
    head = REPO_ROOT / ".git"
    if head.is_file():
        head = Path(head.read_text().partition("gitdir:")[2].strip())
    held = (head / "HEAD").read_text().strip() if (head / "HEAD").exists() else ""
    branch = held.rpartition("/")[2] if held.startswith("ref:") else held[:7]
    return f"{branch or 'detached'}-{datetime.now():%Y%m%d-%H%M%S}"


def sectioned(split: str, term: str) -> str:
    """Return one loss term's key, its own name giving the section it is panelled in.

    Args:
        split: Which split it was measured over.
        term: The term, as the loss names it.

    Returns:
        key: The term under "<split>_<term>", so umr, cmr, mim and loss each
            panel on their own and an instrument sits inside its term's panel.
    """
    named, _, instrument = term.partition("/")
    return f"{split}_{named}" + (f"/{instrument}" if instrument else "")


def start_run(config: Config, facts: Mapping[str, object]) -> Run:
    """Return the tracked run every metric of one training is logged to.

    Args:
        config: What the run reads, trains and how, which the run is recorded
            with.
        facts: What only the assembled run knows, recorded beside the config:
            the device, the parameter count, the patch shapes and how many
            features each split holds.

    Returns:
        run: The run, named for its branch and start, to log to and to finish.
            Which entity and project it
            lands under, and the key it presents, are read from the
            environment, or from the `.env` beside the repository on a run
            here: WANDB_ENTITY, WANDB_PROJECT and WANDB_API_KEY.
    """
    load_dotenv(REPO_ROOT / ".env")
    run = wandb.init(config=asdict(config), name=run_name())
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
        {sectioned("train", name): value.item() for name, value in terms.items()},
        step=step,
    )


def log_epoch(run: Run, step: int, epoch: int, metrics: Mapping[str, float]) -> None:
    """Log one epoch's validation.

    Args:
        run: The tracked run.
        step: Which step of the whole run the epoch ended on.
        epoch: Which pass over the training split it was.
        metrics: Every loss term and retrieval metric over the validation
            split, keyed as they name them.
    """
    logged = {sectioned("validation", name): value for name, value in metrics.items()}
    run.log(logged | {"epoch": epoch}, step=step)


def log_summary(run: Run, summary: Mapping[str, object]) -> None:
    """Write what the whole run came to.

    Args:
        run: The tracked run.
        summary: What it came to, keyed as it is recorded.
    """
    run.summary.update(dict(summary))
