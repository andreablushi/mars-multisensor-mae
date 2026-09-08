"""One sample's crops, fetched from the store and cut into the patches they hold."""

from __future__ import annotations

from dataset.models.patch import Patch
from dataset.models.sample import Sample
from dataset.models.settings import Settings
from dataset.patches import cut
from dataset.store import decode
from dataset.store.artifact import Build


def load_patches(
    sample: Sample, build: Build, settings: Settings, epoch: int = 0
) -> list[Patch]:
    """Return the patches drawn from the observations one sample drew.

    Args:
        sample: The feature and the observations of it that were drawn, which
            name the only crops this fetches.
        build: The published build the crops are read from.
        settings: The settled choices, which say how each crop is cut.
        epoch: Which pass over the dataset this is, so one crop is cut
            differently from one pass to the next.

    Returns:
        patches: The patches of every drawn observation, in the order they were
            drawn and then the order each crop was drawn in.
    """
    return [
        one
        for record in sample.observations
        for one in cut.cut_patches(
            decode.read_crop(build.read(record.path)), record, settings, epoch
        )
    ]
