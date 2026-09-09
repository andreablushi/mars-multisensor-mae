"""The published build in the platform's store, and the keys its crops sit at."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import digitalhub as dh
from botocore.exceptions import ClientError
from shared.disk.files import atomic_path

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
        cache_root: Where the run keeps the crops it has already read, so a
            later pass over them asks the platform for none of them.
    """

    bucket: str
    prefix: str
    client: Any
    cache_root: Path

    def read_object(self, path: str) -> bytes:
        """Return what one object of the build holds, fetching it only once.

        Args:
            path: Where it sits, relative to the build's own root, as the index
                names it.

        Returns:
            data: The bytes of that object.

        Raises:
            ClientError: When the store refused the read for any reason other
                than credentials it had already handed out running out.
        """
        held = self.cache_root / path
        # What one pass fetched every later pass reads off the run's own disk.
        if held.is_file():
            return held.read_bytes()
        key = self.prefix + path
        try:
            fetched = self.client.get_object(Bucket=self.bucket, Key=key)
        except ClientError as refused:
            if refused.response["Error"]["Code"] not in EXPIRED:
                raise
            # The credentials the platform handed out run out mid run.
            self.client = dh.get_s3_client()
            fetched = self.client.get_object(Bucket=self.bucket, Key=key)
        data = fetched["Body"].read()
        # Staged under a name of its own, so a killed run leaves no half object.
        with atomic_path(held) as staged:
            staged.write_bytes(data)
        return data
