"""Publishing what a run produced on DigitalHub, and what it is published as."""

from __future__ import annotations

from pathlib import Path

from dhub import credentials


def published_name(output: str, run_name: str) -> str:
    """Return what one output of a run is published as.

    Args:
        output: What is published: dataset, model or results.
        run_name: What the run is called, or the build for a dataset.

    Returns:
        name: That name, which a later publication versions rather than replaces.
    """
    return f"{output}-{run_name}"


def publish_checkpoint(project, path: Path, run_name: str):
    """Return one checkpoint published as a model of the project.

    Args:
        project: The DigitalHub project the model is logged into.
        path: The checkpoint, on this machine.
        run_name: What the run is called, which names the model.

    Returns:
        model: The logged model.
    """
    credentials.refresh()
    name = published_name("model", run_name)
    return project.log_model(name=name, kind="model", source=str(path))


def publish_results(project, path: Path, run_name: str):
    """Return one evaluation's results published as an artifact of the project.

    Args:
        project: The DigitalHub project the results are logged into.
        path: The results file, on this machine.
        run_name: What the evaluated run is called, which names the artifact.

    Returns:
        artifact: The logged artifact.
    """
    credentials.refresh()
    name = published_name("results", run_name)
    return project.log_artifact(name=name, kind="artifact", source=str(path))
