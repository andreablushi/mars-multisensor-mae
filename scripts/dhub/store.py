"""Reading what DigitalHub published: a build of the dataset, a model, or results."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import digitalhub as dh
from botocore.exceptions import ClientError, ResponseStreamingError

from configs.paths import RESULTS_ROOT, build_root
from dataset.store import DatasetBuild
from dhub import credentials
from dhub.configs import load_platform
from dhub.publish import published_name
from evaluation.results import RESULTS_FILE


def published_build(build: str, root: str) -> DatasetBuild:
    """Return one published build of the dataset, read from the platform's store.

    Args:
        build: The build to read, which is published as dataset-<build>.
        root: Where every build sits here, relative to the repository.

    Returns:
        build: The build, off disk where already fetched and from the store otherwise.
    """
    project = dh.get_or_create_project(load_platform().project)
    published = urlparse(
        project.get_artifact(published_name("dataset", build)).spec.path
    )
    bucket = published.netloc
    prefix = published.path.lstrip("/").rstrip("/") + "/"
    client = dh.get_s3_client()

    def fetch(path: str) -> bytes:
        """Return what one object of the build holds, straight from the store."""
        nonlocal client
        key = prefix + path
        try:
            return client.get_object(Bucket=bucket, Key=key)["Body"].read()
        except (ClientError, ResponseStreamingError):
            # The store's credentials lapse and its streams break mid run.
            credentials.refresh()
            client = dh.get_s3_client()
            try:
                return client.get_object(Bucket=bucket, Key=key)["Body"].read()
            except (ClientError, ResponseStreamingError) as refused:
                raise RuntimeError(f"{key}: {refused}") from None

    return DatasetBuild(root=build_root(build, root), fetch=fetch)


def published_checkpoint(name: str, destination: Path) -> Path:
    """Return where one published model landed on this machine, fetched again each time.

    Args:
        name: What it was published as, the name meaning its latest version.
        destination: The file to write it to, whose directory is made if it is missing.

    Returns:
        path: The checkpoint, on this machine.
    """
    credentials.refresh()
    project = dh.get_or_create_project(load_platform().project)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(project.get_model(name).download(str(destination), overwrite=True))


def fetched_results() -> list[Path]:
    """Return where every published evaluation this machine lacked was fetched to.

    Returns:
        paths: One results file per run fetched, none where it was already here.
    """
    credentials.refresh()
    project = dh.get_or_create_project(load_platform().project)
    prefix = published_name("results", "")
    paths = []
    for artifact in project.list_artifacts():
        held = RESULTS_ROOT / artifact.name.removeprefix(prefix) / RESULTS_FILE
        if not artifact.name.startswith(prefix) or held.is_file():
            continue
        held.parent.mkdir(parents=True, exist_ok=True)
        paths.append(Path(artifact.download(str(held), overwrite=True)))
    return paths
