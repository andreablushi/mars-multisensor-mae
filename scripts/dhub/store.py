"""Reading what DigitalHub published, a build of the dataset or a trained model."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import digitalhub as dh
from botocore.exceptions import ClientError

from config.paths import build_root
from config.schema import DatasetConfig
from dataset.store import DatasetBuild
from dhub import credentials
from dhub.configs import load_platform


def published_build(dataset: DatasetConfig) -> DatasetBuild:
    """Return the build the config names, read from the platform's store.

    Args:
        dataset: What a run reads, naming the build and where it lands here.

    Returns:
        build: The build, off disk where already fetched and from the store otherwise.
    """
    platform = load_platform()
    project = dh.get_or_create_project(platform.project)
    name = f"{platform.publishes['dataset']}-{dataset.build}"
    published = urlparse(project.get_artifact(name).spec.path)
    bucket = published.netloc
    prefix = published.path.lstrip("/").rstrip("/") + "/"
    client = dh.get_s3_client()

    def fetch(path: str) -> bytes:
        """Return what one object of the build holds, straight from the store."""
        nonlocal client
        key = prefix + path
        try:
            fetched = client.get_object(Bucket=bucket, Key=key)
        except ClientError:
            # The store's credentials lapse mid run, and are minted again off the PAT.
            credentials.refresh()
            client = dh.get_s3_client()
            try:
                fetched = client.get_object(Bucket=bucket, Key=key)
            except ClientError as refused:
                raise RuntimeError(f"{key}: {refused}") from None
        return fetched["Body"].read()

    return DatasetBuild(root=build_root(dataset), fetch=fetch)


def published_checkpoint(name: str, destination: Path) -> Path:
    """Return where one published model landed on this machine, fetched again each time.

    Args:
        name: What it was published as, or one version's key, the name meaning latest.
        destination: The file to write it to, whose directory is made if it is missing.

    Returns:
        path: The checkpoint, on this machine.
    """
    credentials.refresh()
    project = dh.get_or_create_project(load_platform().project)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(project.get_model(name).download(str(destination), overwrite=True))
