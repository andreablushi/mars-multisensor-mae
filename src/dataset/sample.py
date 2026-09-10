"""Drawing a few patches of one instrument at random over one feature."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from building.metadata.observation import ObservationMetadata

from dataset.models.patch import Patch
from dataset.patches import cut_patch, patch_counts
from dataset.store import DatasetBuild


def draw_patches(
    observations: Sequence[ObservationMetadata],
    build: DatasetBuild,
    patchsize: dict[str, int],
    count: int,
    rng: np.random.Generator,
) -> list[tuple[ObservationMetadata, Patch]]:
    """Return patches of one instrument drawn at random over one feature.

    Args:
        observations: The feature's index rows of that one instrument.
        build: The published build the observations are read from.
        patchsize: How far a patch runs along every axis it is cut on, by
            instrument, and under "default" for every instrument unnamed.
        count: How many to draw, or every one where the feature holds fewer.
        rng: What fixes the draw.

    Returns:
        drawn: Each patch beside the row it was cut from, every whole patch of
            every observation equally likely and none twice. An observation is
            read only when a patch of it was drawn. Empty where none holds one.
    """
    totals = [
        math.prod(
            patch_counts(
                one.shape, one.axes, patchsize.get(one.instrument, patchsize["default"])
            )
        )
        for one in observations
    ]
    edges = np.cumsum([0, *totals])
    chosen = np.sort(rng.choice(edges[-1], size=min(count, edges[-1]), replace=False))
    drawn = []
    for at, record in enumerate(observations):
        taken = chosen[(chosen >= edges[at]) & (chosen < edges[at + 1])] - edges[at]
        if taken.size == 0:
            continue
        observation = build.read_observation(record.path)
        cut_by = patchsize.get(record.instrument, patchsize["default"])
        drawn.extend(
            (record, cut_patch(observation, record, int(one), cut_by)) for one in taken
        )
    return drawn
