"""Publishing what a run produced on DigitalHub, and what it is published as."""

from __future__ import annotations

from pathlib import Path

from dhub import credentials
from dhub.configs import load_platform

KIND = "model"


def model_name(run_name: str) -> str:
    """Return what a run's best checkpoint is published as.

    Args:
        run_name: What the run is called, as the config names it.

    Returns:
        name: That name, which a later publication versions rather than replaces.
    """
    return f"{load_platform().publishes['model']}-{run_name}"


def publish_checkpoint(project, path: Path, name: str):
    """Return one checkpoint published as a model of the project.

    Args:
        project: The DigitalHub project the model is logged into.
        path: The checkpoint, on this machine.
        name: What to publish it as, which a later one versions rather than replaces.

    Returns:
        model: The logged model.
    """
    credentials.refresh()
    return project.log_model(name=name, kind=KIND, source=str(path))
