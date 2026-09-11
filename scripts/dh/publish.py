"""Publishing what a run produced on DigitalHub, and what it is published as."""

from __future__ import annotations

from pathlib import Path

from config.schema import ModelConfig
from dh import credentials
from dh.configs import load_platform

KIND = "model"


def model_name(model: ModelConfig) -> str:
    """Return what a run's best checkpoint is published as.

    Args:
        model: What the run trained, whose architecture names the publication.

    Returns:
        name: That name, which a later publication of the same one versions
            rather than replaces.
    """
    return f"{load_platform().publishes['model']}-{model.name}"


def publish_checkpoint(project, path: Path, name: str):
    """Return one checkpoint published as a model of the project.

    Args:
        project: The DigitalHub project the model is logged into.
        path: The checkpoint, on this machine.
        name: What to publish it as, which a later publication of the same name
            versions rather than replaces.

    Returns:
        model: The logged model.
    """
    credentials.refresh()
    return project.log_model(name=name, kind=KIND, source=str(path))
