"""The results an evaluation keeps: how far every labelled tile stands from each."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
from common.disk import parquet

RESULTS_FILE = "results.parquet"

RESULTS_SCHEMA = pa.schema(
    [
        ("tile", pa.string()),
        ("label", pa.string()),
        ("distances", pa.list_(pa.float64())),
    ]
)


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
    parquet.write(
        {
            "tile": list(tiles),
            "label": [classes[tile] for tile in tiles],
            "distances": distances.tolist(),
        },
        RESULTS_SCHEMA,
        path,
    )
