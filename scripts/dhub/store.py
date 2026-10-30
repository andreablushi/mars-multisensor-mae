"""What a run reads from DigitalHub: the published builds of the dataset."""

from __future__ import annotations

from urllib.parse import urlparse

import digitalhub as dh
from botocore.exceptions import ClientError, ResponseStreamingError
from urllib3.exceptions import SSLError

from configs.paths import build_root
from configs.schema import Config
from dataset.store import DatasetBuild
from dhub.submit import load_platform

BROKEN_READS = (ClientError, ResponseStreamingError, SSLError)


def published_dataset_build(config: Config, build: str) -> DatasetBuild:
    """Return one published build of the dataset, read from the platform's store.

    Args:
        config: The run's config.
        build: The build to read, which is published as dataset-<build>.

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
            dh.refresh_token()
            client = dh.get_s3_client()
            try:
                return client.get_object(Bucket=bucket, Key=key)["Body"].read()
            except BROKEN_READS as refused:
                raise RuntimeError(f"{key}: {refused}") from None

    return DatasetBuild(
        root=build_root(build, config.dataset.root),
        fetch=fetch,
        clips=config.dataset.clip,
        sharad_atmosphere_cropping=config.dataset.sharad_atmosphere_cropping,
        acquiring=config.training.acquisition,
        acquisition=config.dataset.acquisition,
    )
