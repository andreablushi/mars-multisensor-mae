"""Publishing what a run produced on DigitalHub."""

from __future__ import annotations

from pathlib import Path

from dh import credentials

KIND = "model"


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
