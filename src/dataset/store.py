"""The published build of the dataset, and what its objects hold."""

from __future__ import annotations

import io
import json
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from building import paths as built
from building.common.layout import GROUND
from building.metadata.observation import ObservationMetadata
from building.preprocessing.common.store import EAST, INSIDE, META, NORTH, VALID, native
from shared.disk import parquet
from shared.disk.files import atomic_path

from dataset.models.crop import Crop


@dataclass(slots=True)
class Build:
    """One published build of the dataset, read one object at a time.

    Attributes:
        root: Where the build sits on this machine. A build brought down whole
            fills it, and a build read from elsewhere fills it as it is read,
            so a later pass over the same crops fetches none of them.
        fetch: How one object is brought down when the root holds none of it,
            and None for a build that is already whole on disk.
    """

    root: Path
    fetch: Callable[[str], bytes] | None = None

    def read_object(self, path: str) -> bytes:
        """Return what one object of the build holds, fetching it only once.

        Args:
            path: Where it sits, relative to the build's own root, as the index
                names it.

        Returns:
            data: The bytes of that object.

        Raises:
            FileNotFoundError: When the root holds no such object and the build
                was given nowhere to fetch it from.
        """
        held = self.root / path
        # What one pass fetched every later pass reads off the run's own disk.
        if held.is_file():
            return held.read_bytes()
        if self.fetch is None:
            raise FileNotFoundError(held)
        data = self.fetch(path)
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
