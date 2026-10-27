"""What a run reads from DigitalHub: the published builds of the dataset."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from urllib.parse import urlparse

import digitalhub as dh
from botocore.exceptions import ClientError, ResponseStreamingError
from digitalhub.stores.client.base.factory import get_client
from urllib3.exceptions import SSLError

from configs.paths import build_root
from dataset.store import DatasetBuild
from dhub.submit import load_platform

BROKEN_READS = (ClientError, ResponseStreamingError, SSLError)


def published_build(
    build: str,
    root: str,
    clips: Mapping[str, Sequence[float | None]],
    conditioned: bool,
    acquiring: bool,
    acquisition: Mapping[str, Sequence[str]],
) -> DatasetBuild:
    """Return one published build of the dataset, read from the platform's store.

    Args:
        build: The build to read, which is published as dataset-<build>.
        root: Where every build sits here, relative to the repository.
        clips: The low and high each instrument's normalised values are clipped to.
        conditioned: Whether patches are read with what they are conditioned on.
        acquiring: Whether patches carry the acquisition angles.
        acquisition: The acquisition angles each instrument's patches may carry.

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
            get_client().eval_retry()
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
        acquiring=acquiring,
        acquisition=acquisition,
    )
