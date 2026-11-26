"""What a run reads from DigitalHub: the objects of the published builds."""

from __future__ import annotations

import fcntl
import tempfile
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

import digitalhub as dh
from botocore.exceptions import ClientError, ResponseStreamingError
from urllib3.exceptions import SSLError

from mars_multisensor_mae.configs.schema import Config

BROKEN_READS = (ClientError, ResponseStreamingError, SSLError)
REFRESH_LOCK = Path(tempfile.gettempdir()) / "dhcore-refresh.lock"


def published_fetch(
    config: Config, name: str, subfolder: str = ""
) -> Callable[[str], bytes]:
    """Return how one object of a published build is read from the platform's store.

    Args:
        config: The run's config.
        name: The artifact the build is published in.
        subfolder: The folder of the artifact the build sits in, or "" for its top.

    Returns:
        fetch: What reads one object, by its path relative to the build.
    """
    project = dh.get_or_create_project(config.digitalhub.project)
    published = urlparse(project.get_artifact(name).spec.path)
    bucket = published.netloc
    held_in = f"{subfolder}/" if subfolder else ""
    prefix = published.path.strip("/") + "/" + held_in
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
            with REFRESH_LOCK.open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                dh.refresh_token()
                client = dh.get_s3_client()
            try:
                return client.get_object(Bucket=bucket, Key=key)["Body"].read()
            except BROKEN_READS as refused:
                raise RuntimeError(f"{key}: {refused}") from None

    return fetch
