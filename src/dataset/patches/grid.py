"""How many patches one observation holds and how far each runs, from its index row."""

from __future__ import annotations

import math
from collections.abc import Sequence

from building.common.layout import WAVELENGTH


def patch_lengths(
    shape: Sequence[int], axes: Sequence[str], patchsize: int
) -> tuple[int, ...]:
    """Return how far one patch runs along each axis of an observation.

    Args:
        shape: How many samples each axis of the values holds.
        axes: What each of those axes holds, in that same order.
        patchsize: How far a patch of this instrument runs along an axis it is
            cut on.

    Returns:
        lengths: One length per axis, in the axes' own order. A wavelength axis
            is kept whole, which is how its bands become the channels of every
            patch cut from the observation.
    """
    return tuple(
        size if holds == WAVELENGTH else patchsize
        for size, holds in zip(shape, axes, strict=True)
    )


def patch_counts(
    shape: Sequence[int], axes: Sequence[str], patchsize: int
) -> tuple[int, ...]:
    """Return how many patches fit along each axis of an observation.

    Args:
        shape: How many samples each axis of the values holds.
        axes: What each of those axes holds, in that same order.
        patchsize: How far a patch of this instrument runs along an axis it is
            cut on.

    Returns:
        counts: How many whole patches each axis holds. What is left of an axis
            after the last whole one is dropped rather than padded, so an axis
            shorter than one patch holds none and the observation holds none.
    """
    return tuple(
        size // length
        for size, length in zip(
            shape, patch_lengths(shape, axes, patchsize), strict=True
        )
    )


def patch_count(shape: Sequence[int], axes: Sequence[str], patchsize: int) -> int:
    """Return how many patches one observation holds, without reading it.

    Args:
        shape: How many samples each axis of the values holds.
        axes: What each of those axes holds, in that same order.
        patchsize: How far a patch of this instrument runs along an axis it is
            cut on.

    Returns:
        counted: How many whole patches it holds.
    """
    return math.prod(patch_counts(shape, axes, patchsize))
