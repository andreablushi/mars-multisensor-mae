"""What a patch of one instrument goes through before the model reads it."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

import numpy as np
from building.common.layout import GROUND

from dataset.models.patch import Patch


def pooled_patch(patch: Patch, pool: Mapping[str, int]) -> Patch:
    """Return a patch as the model reads its instrument.

    Args:
        patch: The patch, as cut from its observation.
        pool: How many ground samples of a patch each instrument averages into one.

    Returns:
        patch: The patch, pooled along the ground where its instrument asks for it.
    """
    factor = pool.get(patch.instrument, 1)
    if factor == 1:
        return patch

    def split(shape: Sequence[int]) -> tuple[int, ...]:
        return tuple(
            length
            for size, holds in zip(shape, patch.axes, strict=True)
            for length in ((size // factor, factor) if holds == GROUND else (size,))
        )

    # Where each ground axis's pooled samples land once it is split in two
    ends = np.cumsum([2 if holds == GROUND else 1 for holds in patch.axes]) - 1
    pooled = tuple(
        int(end) for end, holds in zip(ends, patch.axes, strict=True) if holds == GROUND
    )
    valid = patch.valid.reshape(split(patch.valid.shape))
    weight = np.broadcast_to(valid, split(patch.values.shape)).astype(np.float32)
    values = np.where(weight > 0, patch.values.reshape(weight.shape), 0.0)
    counted = weight.sum(axis=pooled)
    return dataclasses.replace(
        patch,
        values=(values * weight).sum(axis=pooled) / np.maximum(counted, 1.0),
        valid=valid.any(axis=pooled),
    )


def pooled_patch_shape(
    shape: Sequence[int], axes: Sequence[str], factor: int
) -> tuple[int, ...]:
    """Return the shape of one patch as the model reads it.

    Args:
        shape: The shape of one patch as cut from its observation.
        axes: What each of its axes holds, in that same order.
        factor: How many ground samples its instrument averages into one.

    Returns:
        shape: The shape once pooled along the ground.
    """
    return tuple(
        size // factor if holds == GROUND else size
        for size, holds in zip(shape, axes, strict=True)
    )
