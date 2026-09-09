"""The published build in the platform's store, and what its objects hold."""

from __future__ import annotations

import io
import json
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import digitalhub as dh
import numpy as np
import pyarrow.parquet as pq
from botocore.exceptions import ClientError
from building import paths as built
from building.common.layout import GROUND
from building.metadata.observation import ObservationMetadata
from building.preprocessing.common.store import EAST, INSIDE, META, NORTH, VALID, native
from shared.disk import parquet
from shared.disk.files import atomic_path

from dataset.models.crop import Crop

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


def read_build_manifest(build: Build) -> dict:
    """Return what the build says it is, as it wrote it.

    Args:
        build: The published build to read.

    Returns:
        manifest: When it was built, from what, and which instruments it reached.
    """
    return json.loads(build.read_object(built.DATASET_MANIFEST_NAME))


def read_observation_metadata(build: Build) -> list[ObservationMetadata]:
    """Return what every crop of the build is, without reading one of them.

    Args:
        build: The published build to read.

    Returns:
        records: One row per crop, in the order the index holds them. The file
            is read under no schema, so a build carrying columns the row model
            does not declare is read all the same.
    """
    held = pq.read_table(io.BytesIO(build.read_object(built.OBSERVATION_METADATA_NAME)))
    return [parquet.build(ObservationMetadata, row) for row in held.to_pylist()]


def observation_metadata_by_feature(
    observations: Iterable[ObservationMetadata],
) -> dict[tuple[str, str], list[ObservationMetadata]]:
    """Return the crops of each feature, keyed by what tells that feature apart.

    Args:
        observations: The index rows to group, in any order.

    Returns:
        standing: The rows of each feature, in the order the index held them.
    """
    standing: dict[tuple[str, str], list[ObservationMetadata]] = defaultdict(list)
    for one in observations:
        standing[one.feature].append(one)
    return dict(standing)


def drawn_observation_metadata(
    observations: Sequence[ObservationMetadata], config: dict
) -> list[ObservationMetadata]:
    """Return a draw of the crops the build holds of one feature.

    Args:
        observations: One feature's own index rows, which the draw is made from.
        config: The choices a read is made with, which say how many of each
            instrument a sample draws and what number fixes that draw.

    Returns:
        drawn: At most as many of each instrument as the config asks for, the
            instruments in name order. The draw is fixed by the feature's own
            name, so every process draws the same crops of it.
    """
    standing: dict[str, list[ObservationMetadata]] = defaultdict(list)
    for one in observations:
        standing[one.instrument].append(one)
    drawing = random.Random(f"{config['seed']}/{'/'.join(observations[0].feature)}")
    return [
        one
        for instrument in sorted(standing)
        for one in drawing.sample(
            standing[instrument],
            min(config["per_instrument"], len(standing[instrument])),
        )
    ]


def read_crop(data: bytes) -> Crop:
    """Return one stored observation, read out of the bytes it was written as.

    Args:
        data: What the crop's own object holds.

    Returns:
        crop: The observation, its values in the machine's byte order and every
            sample counted as measured where the build stored no mask.
    """
    with np.load(io.BytesIO(data)) as held:
        # What the crop is, is stored beside its arrays as one json string.
        described = json.loads(str(held[META]))
        arrays = {name: native(held[name]) for name in held.files if name != META}
    axes = tuple(described["axes"])
    values = arrays.pop(described["measurement"])
    # The masks run over the ground axes alone, not over bands or depth.
    ground = tuple(
        size for size, holds in zip(values.shape, axes, strict=True) if holds == GROUND
    )
    # A mask the build stored none of marked every sample, so one stands in.
    measured = np.ones(ground, dtype=bool)
    for name in (VALID, INSIDE):
        mask = arrays.pop(name, None)
        if mask is not None:
            measured &= mask
    polar = described["polar"]
    return Crop(
        instrument=described["instrument"],
        identifier=described["identifier"],
        measurement=described["measurement"],
        values=values,
        axes=axes,
        dims={name: tuple(held) for name, held in described["dims"].items()},
        measured=measured,
        north=arrays.pop(NORTH),
        east=arrays.pop(EAST),
        position_units=described["position_units"],
        polar=None if polar is None else tuple(polar),
        centre_lon=described["centre_lon"],
        centre_lat=described["centre_lat"],
        # What the pops left is what the instrument stores beside its values.
        beside=arrays,
    )
