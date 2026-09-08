"""The published build in the platform's store, and the keys its crops sit at."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import digitalhub as dh
from botocore.exceptions import ClientError

from dataset import configs
from dataset.models.settings import Settings
from dataset.store.cache import Cache

EXPIRED = frozenset(
    {"ExpiredToken", "ExpiredTokenException", "InvalidToken", "InvalidAccessKeyId"}
)


@dataclass(slots=True)
class Build:
    """One published build of the dataset, read one object at a time.

    Attributes:
        bucket: The bucket the platform published it in.
        prefix: The key every path the index names hangs off.
        client: The client it is read with, minted again when the credentials
            behind it run out.
        cache: What the run keeps of the crops it has already read.
    """

    bucket: str
    prefix: str
    client: Any
    cache: Cache

    def read(self, path: str) -> bytes:
        """Return what one object of the build holds, fetching it only once.

        Args:
            path: Where it sits, relative to the build's own root, as the index
                names it.

        Returns:
            data: The bytes of that object.
        """
        held = self.cache.kept(path)
        if held is not None:
            return held
        data = self._fetched(path)
        self.cache.keep(path, data)
        return data

    def _fetched(self, path: str) -> bytes:
        """Return one object of the build, minting a client again where one ran out.

        Args:
            path: Where it sits, relative to the build's own root.

        Returns:
            data: The bytes of that object.

        Raises:
            ClientError: When the store refused the read for any reason other
                than credentials it had already handed out running out.
        """
        try:
            return self._object(path)
        except ClientError as refused:
            if refused.response["Error"]["Code"] not in EXPIRED:
                raise
            self.client = dh.get_s3_client()
            return self._object(path)

    def _object(self, path: str) -> bytes:
        """Return one object of the build, asking the store for it once.

        Args:
            path: Where it sits, relative to the build's own root.

        Returns:
            data: The bytes of that object.
        """
        held = self.client.get_object(Bucket=self.bucket, Key=self.prefix + path)
        return held["Body"].read()


def opened_build(settings: Settings) -> Build:
    """Return the published build one read is made against.

    Args:
        settings: The settled choices, which name the project it belongs to and
            the build to read.

    Returns:
        build: The build, its prefix resolved off the platform and its cache
            ready to take what it reads.

    Raises:
        ValueError: When the platform publishes the build somewhere other than
            the object store this reads from.
    """
    project = dh.get_project(settings.project)
    published = urlparse(project.get_artifact(settings.artifact).spec.path)
    if published.scheme != "s3":
        raise ValueError(f"{settings.artifact} is published at {published.scheme}")
    prefix = published.path.lstrip("/")
    return Build(
        bucket=published.netloc,
        prefix=prefix if prefix.endswith("/") else f"{prefix}/",
        client=dh.get_s3_client(),
        cache=Cache(root=configs.cache_root(), budget=settings.cache_bytes),
    )
