"""Reading a published build out of the store DigitalHub keeps it in."""

from __future__ import annotations

from urllib.parse import urlparse

import digitalhub as dh
from botocore.exceptions import ClientError

from config.paths import build_root
from config.schema import DatasetConfig
from dataset.store import DatasetBuild
from dh import credentials
from dh.configs import load_platform

EXPIRED = frozenset(
    {"ExpiredToken", "ExpiredTokenException", "InvalidToken", "InvalidAccessKeyId"}
)


def published_build(dataset: DatasetConfig) -> DatasetBuild:
    """Return the build the config names, read from the platform's store.

    Args:
        dataset: What a run reads, which names the build to read and where it
            lands on this machine.

    Returns:
        build: The build, reading off disk every observation it has already fetched and
            asking the store for the rest.
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
        except ClientError as refused:
            if refused.response["Error"]["Code"] not in EXPIRED:
                raise
            # The credentials the platform hands out run out mid run.
            credentials.refresh()
            client = dh.get_s3_client()
            fetched = client.get_object(Bucket=bucket, Key=key)
        return fetched["Body"].read()

    return DatasetBuild(root=build_root(dataset), fetch=fetch)
