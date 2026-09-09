"""The crops one sample drew, fetched from the store and cut into their patches."""

from __future__ import annotations

from collections.abc import Iterable

from building.metadata.observation import ObservationMetadata

from dataset.models.patch import Patch
from dataset.patches import cut
from dataset.store import decode
from dataset.store.read_dh_volumes import Build


def load_patches(
    observations: Iterable[ObservationMetadata], build: Build, config: dict
) -> list[Patch]:
    """Return the patches drawn from the observations a sample drew.

    Args:
        observations: The index rows that were drawn, which name the only crops
            this fetches.
        build: The published build the crops are read from.
        config: The choices a read is made with, which say how each crop is cut.

    Returns:
        patches: The patches of every drawn observation, in the order they were
            drawn and then the order each crop was cut in.
    """
    return [
        one
        for record in observations
        for one in cut.cut_patches(
            decode.read_crop(build.read_object(record.path)), record, config
        )
    ]
