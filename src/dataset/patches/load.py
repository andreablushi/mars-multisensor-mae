"""Every crop of one feature, fetched from the store and cut into its patches."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from building.metadata.observation import ObservationMetadata

from dataset.models.patch import Patch
from dataset.patches.cut import cut_patches
from dataset.store import Build, read_crop


def load_patches(
    observations: Iterable[ObservationMetadata], build: Build, config: dict
) -> Iterator[Patch]:
    """Yield every patch of every observation of one feature.

    Args:
        observations: The feature's own index rows, every crop the build holds
            of it.
        build: The published build the crops are read from.
        config: The choices a read is made with, which say how each crop is cut.

    Yields:
        patch: Every patch of every crop, one crop at a time, so only the crop
            being cut is held in memory. A feature runs to over a million.
    """
    for record in observations:
        yield from cut_patches(
            read_crop(build.read_object(record.path)), record, config
        )
