"""How many patches one observation holds and how far each runs, from its index row."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def patch_lengths(
    shape: Sequence[int], axes: Sequence[str], tiles: Mapping[str, int]
) -> tuple[int, ...]:
    """Return how far one patch runs along each axis of an observation.

    Args:
        shape: How many samples each axis of the values holds.
        axes: What each of those axes holds, in that same order.
        tiles: How far a patch runs along each kind of axis. An axis whose kind
            is not named there is kept whole, which is how a wavelength axis
            becomes the channels of every patch cut from it.

    Returns:
        lengths: One length per axis, in the axes' own order.
    """
    return tuple(
        tiles.get(holds, size) for size, holds in zip(shape, axes, strict=True)
    )


def patch_counts(
    shape: Sequence[int], axes: Sequence[str], tiles: Mapping[str, int]
) -> tuple[int, ...]:
    """Return how many patches fit along each axis of an observation.

    Args:
        shape: How many samples each axis of the values holds.
        axes: What each of those axes holds, in that same order.
        tiles: How far a patch runs along each kind of axis.

    Returns:
        counts: How many whole patches each axis holds. What is left of an axis
            after the last whole one is dropped rather than padded, so an axis
            shorter than one patch holds none and the observation holds none.
    """
    return tuple(
        size // length
        for size, length in zip(shape, patch_lengths(shape, axes, tiles), strict=True)
    )


def patch_count(
    shape: Sequence[int], axes: Sequence[str], tiles: Mapping[str, int]
) -> int:
    """Return how many patches one observation holds, without reading it.

    Args:
        shape: How many samples each axis of the values holds.
        axes: What each of those axes holds, in that same order.
        tiles: How far a patch runs along each kind of axis.

    Returns:
        counted: How many whole patches it holds, before any of them is weighed
            against how much of it was measured.
    """
    return math.prod(patch_counts(shape, axes, tiles))
