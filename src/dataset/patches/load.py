"""Every crop of one feature, fetched from the store and cut into its patches."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from building.metadata.observation import ObservationMetadata

from dataset.models.patch import Patch
from dataset.patches.cut import cut_patches
from dataset.store import DatasetBuild


def load_patches(
    observations: Iterable[ObservationMetadata],
    build: DatasetBuild,
    patchsize: dict[str, int],
) -> Iterator[Patch]:
    """Yield every patch of every observation of one feature.

    Args:
        observations: The feature's own index rows, every crop the build holds
            of it.
        build: The published build the crops are read from.
        patchsize: How far a patch runs along every axis it is cut on, by
            instrument, and under "default" for every instrument unnamed.

    Yields:
        patch: Every patch of every crop, one crop at a time, so only the crop
            being cut is held in memory. A feature runs to over a million.
    """
    for record in observations:
        cut_by = patchsize.get(record.instrument, patchsize["default"])
        yield from cut_patches(build.read_crop(record.path), record, cut_by)
