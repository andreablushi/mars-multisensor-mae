"""Tracking a run on Weights & Biases: its stage, its training curves and its model."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import wandb
from dotenv import load_dotenv
from torch import Tensor
from wandb.sdk.wandb_run import Run

from configs.paths import ARTIFACTS_ROOT, ENV_PATH

MODEL = "model"
TRAINING = "training"
TRAIN = f"{TRAINING}/train"
VALIDATION = f"{TRAINING}/validation"


def start_logging(
    config: Mapping[str, object],
    run_name: str,
    stage: str,
    facts: Mapping[str, object],
) -> Run:
    """Return the tracked run of one stage, replacing any of its name and stage.

    Args:
        config: What the run reads, trains and how, which the run is recorded with.
        run_name: What the run is called, which names the tracked run, the part
            before a slash grouping it.
        stage: What the stage is called, which tells a training from an evaluation.
        facts: What only the assembled run knows, recorded beside the config.

    Returns:
        run: The run, named and grouped by the run name so the training and
            the evaluation of one model sit together, its place read from the .env.
    """
    load_dotenv(ENV_PATH)
    for tracked in wandb.Api().runs(
        f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}",
        filters={"display_name": run_name},
    ):
        if tracked.job_type == stage:
            tracked.delete(delete_artifacts=True)
    group = run_name.split("/")[0]
    run = wandb.init(config=dict(config), name=run_name, group=group, job_type=stage)
    run.config.update(dict(facts))
    return run


def log_terms(
    run: Run, step: int, prefix: str, terms: Mapping[str, Tensor | float]
) -> None:
    """Log one step's terms under a prefix.

    Args:
        run: The tracked run.
        step: Which step of the whole run it is.
        prefix: Where the terms are logged, TRAIN or VALIDATION.
        terms: Every term, keyed as the loss names them.
    """
    run.log(
        {f"{prefix}/{name}": float(value) for name, value in terms.items()}, step=step
    )


def artifact_name(kind: str, run_name: str) -> str:
    """Return what one run's artifact of a kind is called.

    Args:
        kind: What the artifact holds, its type too.
        run_name: What the run is called.

    Returns:
        name: The kind and the run name, its slashes made dots.
    """
    return f"{kind}-{run_name.replace('/', '.')}"


def log_checkpoint(run: Run, path: Path, run_name: str) -> None:
    """Upload one checkpoint as the run's model, waiting until it is stored.

    Args:
        run: The tracked run that trained it.
        path: The checkpoint, on this machine.
        run_name: What the run is called, which names the model.
    """
    model = wandb.Artifact(artifact_name(MODEL, run_name), type=MODEL)
    model.add_file(str(path))
    run.log_artifact(model).wait()


def logged_checkpoint(run: Run, run_name: str) -> Path:
    """Return one run's latest model, downloaded and recorded as the run's input.

    Args:
        run: The tracked run that reads it.
        run_name: What the trained run is called, which names the model.

    Returns:
        path: The checkpoint, on this machine.
    """
    model = run.use_artifact(f"{artifact_name(MODEL, run_name)}:latest")
    return Path(model.file(str(ARTIFACTS_ROOT / model.name)))
