"""What a run reads from DigitalHub and publishes to it: builds, models, results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import digitalhub as dh
from botocore.exceptions import ClientError, ResponseStreamingError
from digitalhub.stores.client.base.factory import get_client
from digitalhub.stores.data.api import get_default_store
from urllib3.exceptions import SSLError

from configs.paths import (
    RECONSTRUCTION_FILE,
    RESULTS_FILE,
    build_root,
    checkpoint_path,
    results_path,
)
from dataset.store import DatasetBuild
from dhub.submit import load_platform

BROKEN_READS = (ClientError, ResponseStreamingError, SSLError)

RESULTS_STORE = "results"

EVALUATION_FILES = {
    Path(name).stem: name for name in (RESULTS_FILE, RECONSTRUCTION_FILE)
}


def refresh_credentials() -> None:
    """Mint the run's credentials again, the platform's and the store's alike."""
    get_client().eval_retry()


def published_build(
    build: str,
    root: str,
    clips: Mapping[str, Sequence[float | None]],
    conditioned: bool,
) -> DatasetBuild:
    """Return one published build of the dataset, read from the platform's store.

    Args:
        build: The build to read, which is published as dataset-<build>.
        root: Where every build sits here, relative to the repository.
        clips: The low and high each instrument's normalised values are clipped to.
        conditioned: Whether patches are read with what they are conditioned on.

    Returns:
        build: The build, off disk where already fetched and from the store otherwise.
    """
    project = dh.get_or_create_project(load_platform()["project"])
    published = urlparse(project.get_artifact(f"dataset-{build}").spec.path)
    bucket = published.netloc
    prefix = published.path.strip("/") + "/"
    client = dh.get_s3_client()

    def fetch(path: str) -> bytes:
        """Return what one object of the build holds, straight from the store.

        Args:
            path: The object, relative to the build.

        Returns:
            data: The object's bytes.

        Raises:
            RuntimeError: When the read breaks again once the credentials are minted.
        """
        nonlocal client
        key = prefix + path
        try:
            return client.get_object(Bucket=bucket, Key=key)["Body"].read()
        except BROKEN_READS:
            # The store's credentials lapse and its streams break mid run.
            refresh_credentials()
            client = dh.get_s3_client()
            try:
                return client.get_object(Bucket=bucket, Key=key)["Body"].read()
            except BROKEN_READS as refused:
                raise RuntimeError(f"{key}: {refused}") from None

    return DatasetBuild(
        root=build_root(build, root),
        fetch=fetch,
        clips=clips,
        conditioned=conditioned,
    )


def published_checkpoint(run_name: str, checkpoints: str) -> Path:
    """Return where one run's latest published model landed here, fetched each time.

    Args:
        run_name: What the run is called, which names the model.
        checkpoints: Where checkpoints land, relative to the repository.

    Returns:
        path: The checkpoint, on this machine.
    """
    refresh_credentials()
    project = dh.get_or_create_project(load_platform()["project"])
    name = f"model-{run_name.replace('/', '.')}"
    destination = checkpoint_path(checkpoints, name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(project.get_model(name).download(str(destination), overwrite=True))


def fetched_results() -> list[Path]:
    """Return where each published evaluation file missing or older here was fetched.

    Returns:
        paths: One file per run and kind fetched.
    """
    refresh_credentials()
    project = dh.get_or_create_project(load_platform()["project"])
    paths = []
    for artifact in project.list_artifacts():
        kind, _, run = artifact.name.partition("-")
        if kind not in EVALUATION_FILES:
            continue
        held = results_path(run.replace(".", "/"), EVALUATION_FILES[kind])
        published = datetime.fromisoformat(artifact.metadata.updated).timestamp()
        if held.is_file() and held.stat().st_mtime >= published:
            continue
        held.parent.mkdir(parents=True, exist_ok=True)
        paths.append(Path(artifact.download(str(held), overwrite=True)))
    return paths


def publish_checkpoint(project, path: Path, run_name: str):
    """Return one checkpoint published as a model of the project.

    Args:
        project: The DigitalHub project the model is logged into.
        path: The checkpoint, on this machine.
        run_name: What the run is called, which names the model.

    Returns:
        model: The logged model.
    """
    refresh_credentials()
    name = f"model-{run_name.replace('/', '.')}"
    return project.log_model(name=name, kind="model", source=str(path))


def publish_results(project, path: Path, run_name: str):
    """Return one evaluation file published as an artifact of the project.

    Args:
        project: The DigitalHub project the file is logged into.
        path: The file, on this machine, whose stem names the artifact's kind.
        run_name: What the evaluated run is called, which names the artifact.

    Returns:
        artifact: The logged artifact, named <stem>-<run_name>, kept under results/.
    """
    refresh_credentials()
    name = f"{path.stem}-{run_name.replace('/', '.')}"
    uuid = uuid4().hex
    store = f"{get_default_store(project.name)}/{project.name}/{RESULTS_STORE}"
    return project.log_artifact(
        name=name,
        kind="artifact",
        source=str(path),
        path=f"{store}/{name}/{uuid}/{path.name}",
        uuid=uuid,
    )
