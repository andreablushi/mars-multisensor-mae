"""Writing what a run does to the terminal, and tracking it on Weights & Biases."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import wandb
from torch import Tensor
from wandb.apis.public import Run as TrackedRun
from wandb.sdk.wandb_run import Run

TRAINING = "training"
TRAIN = f"{TRAINING}/train"
VALIDATION = f"{TRAINING}/validation"

RESULTS_STEP = 10**6


def console_logger(name: str) -> logging.Logger:
    """Return a logger writing to the console.

    Args:
        name: Whose logger, which is the module's own name.

    Returns:
        log: The logger.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    return logging.getLogger(name)


def start_logging(
    config: Mapping[str, object],
    run_name: str,
    stage: str,
    facts: Mapping[str, object],
    replaced: Sequence[str] = (),
    spared: str | None = None,
) -> Run:
    """Return the tracked run of one stage, replacing any of its name and stage.

    Args:
        config: What the run reads, trains and how, which the run is recorded with.
        run_name: What the run is called, which names the tracked run, the part
            before a slash grouping it.
        stage: What the stage is called, which tells a training from an evaluation.
        facts: What only the assembled run knows, recorded beside the config.
        replaced: The other stages whose runs of the name are deleted too.
        spared: The id of one run of the name left alone, for its caller to retire.

    Returns:
        run: The run, named and grouped by the run name so the training and
            the evaluation of one model sit together, its place read from the
            environment.
    """
    for tracked in wandb.Api().runs(
        f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}",
        filters={"display_name": run_name},
    ):
        if tracked.job_type in (stage, *replaced) and tracked.id != spared:
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


def flat_run_name(run_name: str) -> str:
    """Return the run name with its slashes made dots, for names that refuse slashes.

    Args:
        run_name: What the run is called.

    Returns:
        name: The same name, its slashes made dots.
    """
    return run_name.replace("/", ".")


def artifact_name(kind: str, run_name: str) -> str:
    """Return what one run's artifact of a kind is called.

    Args:
        kind: What the artifact holds, its type too.
        run_name: What the run is called.

    Returns:
        name: The kind and the run name, its slashes made dots.
    """
    return f"{kind}-{flat_run_name(run_name)}"


def log_checkpoint(run: Run, path: Path, run_name: str, kind: str = "model") -> None:
    """Upload one checkpoint as the run's artifact of a kind, waiting until stored.

    Args:
        run: The tracked run that trained it.
        path: The checkpoint, on this machine.
        run_name: What the run is called, which names the artifact.
        kind: What the checkpoint holds, the model or the benchmark's heads.
    """
    artifact = wandb.Artifact(artifact_name(kind, run_name), type=kind)
    artifact.add_file(str(path))
    run.log_artifact(artifact).wait()


def tracked_run(run_name: str, stage: str) -> TrackedRun | None:
    """Return the tracked run of one name and stage, as the API reads it, if any.

    Args:
        run_name: What the run is called, which names the tracked run.
        stage: What the stage is called, which tells a training from an evaluation.

    Returns:
        run: The run, or None when no run of the name and stage is tracked.
    """
    for tracked in wandb.Api().runs(
        f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}",
        filters={"display_name": run_name},
    ):
        if tracked.job_type == stage:
            return tracked
    return None


def resumed_logging(
    run_name: str, stages: Sequence[str], kind: str, artifacts_root: Path
) -> tuple[Run, Path]:
    """Return the tracked run that logged a run's checkpoint of a kind, reopened.

    Args:
        run_name: What the run is called, which names the artifact.
        stages: The stages whose other runs of the name are deleted, so one is left.
        kind: What the checkpoint holds, the model or the benchmark's heads.
        artifacts_root: Where downloaded artifacts are kept on this machine.

    Returns:
        run: The run that logged the checkpoint, reopened to log more.
        path: The checkpoint, on this machine.
    """
    api = wandb.Api()
    place = f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}"
    artifact = api.artifact(f"{place}/{artifact_name(kind, run_name)}:latest")
    owner = artifact.logged_by()
    for tracked in api.runs(place, filters={"display_name": run_name}):
        if tracked.job_type in stages and tracked.id != owner.id:
            tracked.delete(delete_artifacts=True)
    path = Path(artifact.file(str(artifacts_root / artifact.name)))
    return wandb.init(id=owner.id, resume="must"), path


def logged_checkpoint(
    run: Run, run_name: str, artifacts_root: Path, kind: str = "model"
) -> Path:
    """Return one run's latest checkpoint of a kind, recorded as the run's input.

    Args:
        run: The tracked run that reads it.
        run_name: What the run that logged it is called, which names the artifact.
        artifacts_root: Where downloaded artifacts are kept on this machine.
        kind: What the checkpoint holds, the model or the benchmark's heads.

    Returns:
        path: The checkpoint, on this machine.
    """
    artifact = run.use_artifact(f"{artifact_name(kind, run_name)}:latest")
    return Path(artifact.file(str(artifacts_root / artifact.name)))
