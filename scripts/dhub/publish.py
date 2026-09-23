"""Publishing what a run produced on DigitalHub, and what it is published as."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from digitalhub.stores.data.api import get_default_store

from dhub import credentials
from dhub.configs import load_platform

KIND = "model"
RESULTS_KIND = "artifact"


def published_name(output: str, run_name: str) -> str:
    """Return what one output of a run is published as.

    Args:
        output: What is published, as the platform config names it: model or results.
        run_name: What the run is called, as the config names it.

    Returns:
        name: That name, which a later publication versions rather than replaces.
    """
    return f"{load_platform().publishes[output]}-{run_name}"


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


def publish_results(project, path: Path, name: str):
    """Return one evaluation's results published as an artifact of the project.

    Args:
        project: The DigitalHub project the results are logged into.
        path: The results file, on this machine.
        name: What to publish it as, which a later one versions rather than replaces.

    Returns:
        artifact: The logged artifact, stored under results/ beside artifacts/.
    """
    credentials.refresh()
    version = uuid4().hex
    store = get_default_store(project.name)
    held = f"{store}/{project.name}/results/{name}/{version}/{path.name}"
    return project.log_artifact(
        name=name, kind=RESULTS_KIND, source=str(path), path=held, uuid=version
    )
