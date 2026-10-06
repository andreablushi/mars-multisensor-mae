"""The results an evaluation keeps: how far labelled tiles stand, per instrument."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from common.disk import parquet


@dataclass(frozen=True, slots=True)
class TileDistances:
    """One labelled tile, its class and its distance to every tile over one instrument.

    Attributes:
        tile: The tile.
        label: The class it earned.
        instrument: The instrument the distances compare, or EVERY_INSTRUMENT for
            their mean.
        distances: Its distance to every tile, in row order.
    """

    tile: str
    label: str
    instrument: str
    distances: tuple[float, ...]


RESULTS_SCHEMA = parquet.schema_of(TileDistances)

EVERY_INSTRUMENT = "all"


def write_tile_distances(
    path: Path,
    tiles: Sequence[str],
    classes: Mapping[str, str],
    distances: Mapping[str, np.ndarray],
) -> None:
    """Write one row per tile and instrument: its class and its distance to every tile.

    Args:
        path: The parquet file to write, whose directory is made if missing.
        tiles: The tiles, in the order the distances hold them.
        classes: The class each tile earned, keyed by the tile.
        distances: The distance between every pair of tiles, keyed by the instrument
            compared or EVERY_INSTRUMENT. (T, T)
    """
    parquet.write_rows(
        [
            TileDistances(tile, classes[tile], instrument, tuple(row))
            for instrument, matrix in distances.items()
            for tile, row in zip(tiles, matrix.tolist(), strict=True)
        ],
        RESULTS_SCHEMA,
        path,
    )
