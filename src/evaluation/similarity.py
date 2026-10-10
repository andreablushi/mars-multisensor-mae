"""How alike tiles are, per instrument and over every instrument together."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

EVERY_INSTRUMENT = "all"


def cosine_similarities(vectors: np.ndarray) -> np.ndarray:
    """Return the cosine between every pair of tiles' vectors of one instrument.

    Args:
        vectors: One unit vector per tile, NaN for a tile without the instrument. (T, D)

    Returns:
        similarities: The cosine of every pair, NaN where either tile lacks it. (T, T)
    """
    return vectors @ vectors.T


def fused_similarities(similarities: Sequence[np.ndarray]) -> np.ndarray:
    """Return every pair's cosine averaged over the instruments both tiles hold.

    Args:
        similarities: One instrument's cosines per entry, NaN where a pair lacks it.
            (T, T)

    Returns:
        fused: The mean over the instruments a pair shares, NaN where it shares none.
            (T, T)
    """
    stacked = np.stack(similarities)  # (M, T, T)
    shared = np.isfinite(stacked).sum(axis=0)  # (T, T)
    summed = np.nansum(stacked, axis=0)  # (T, T)
    return np.where(shared > 0, summed / np.maximum(shared, 1), np.nan)


def similarities_by_view(vectors: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return the cosines of every instrument alone, and fused over all of them.

    Args:
        vectors: Each instrument's unit vectors, NaN for a tile without it. (T, D)

    Returns:
        similarities: The cosine of every pair, keyed by instrument and
            EVERY_INSTRUMENT. (T, T)
    """
    alone = {name: cosine_similarities(held) for name, held in vectors.items()}
    return alone | {EVERY_INSTRUMENT: fused_similarities(list(alone.values()))}


def held_tiles(similarities: np.ndarray) -> np.ndarray:
    """Return which tiles a view holds, those with a vector to compare.

    Args:
        similarities: The cosine of every pair. (T, T)

    Returns:
        held: True where the tile has a similarity to itself. (T,)
    """
    return np.isfinite(np.diag(similarities))
