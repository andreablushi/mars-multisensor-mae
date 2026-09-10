"""Publishing what a run produced on DigitalHub."""

from __future__ import annotations

from pathlib import Path

import digitalhub as dh
from dotenv import load_dotenv

from config.paths import REPO_ROOT
from dh.store import PROJECT

KIND = "artifact"


def publish_checkpoint(path: Path, name: str) -> None:
    """Publish one checkpoint as an artifact of the project.

    Args:
        path: The checkpoint, on this machine.
        name: What to publish it as, which a later publication of the same name
            versions rather than replaces.
    """
    load_dotenv(REPO_ROOT / ".env")
    project = dh.get_or_create_project(PROJECT)
    project.log_artifact(name=name, kind=KIND, source=str(path))
