"""The results an evaluation keeps: how far every labelled tile stands from each."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from common.disk import parquet

RESULTS_FILE = "results.parquet"


@dataclass(frozen=True, slots=True)
class TileDistances:
    """One labelled tile, its class and its distance to every tile.

    Attributes:
        tile: The tile.
        label: The class it earned.
        distances: Its distance to every tile, in row order.
    """

    tile: str
    label: str
    distances: tuple[float, ...]


RESULTS_SCHEMA = parquet.schema_of(TileDistances)


def write_tile_distances(
    path: Path,
    tiles: Sequence[str],
    classes: Mapping[str, str],
    distances: np.ndarray,
) -> None:
    """Write one row per tile: its class and its distance to every tile, in row order.

    Args:
        path: The parquet file to write, whose directory is made if missing.
        tiles: The tiles, in the order the distances hold them.
        classes: The class each tile earned, keyed by the tile.
        distances: The distance between every pair of tiles. (T, T)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_rows(
        [
            TileDistances(tile, classes[tile], tuple(row))
            for tile, row in zip(tiles, distances.tolist(), strict=True)
        ],
        RESULTS_SCHEMA,
        path,
    )
