"""The results an evaluation keeps: one vector per labelled tile and instrument."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from common.disk import parquet


@dataclass(frozen=True, slots=True)
class TileVector:
    """One labelled tile, its class and its vector over one instrument.

    Attributes:
        tile: The tile.
        label: The class it earned.
        weights: Which model read it, TRAINED or RANDOM.
        instrument: The instrument the vector stands for.
        vector: Its tokens averaged, of unit length.
    """

    tile: str
    label: str
    weights: str
    instrument: str
    vector: tuple[float, ...]


RESULTS_SCHEMA = parquet.schema_of(TileVector)

TRAINED = "trained"
RANDOM = "random"


def write_tile_vectors(
    path: Path,
    classes: Mapping[str, str],
    vectors: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
) -> None:
    """Write one row per model, instrument and tile: its class and its vector.

    Args:
        path: The parquet file to write, whose directory is made if missing.
        classes: The class each tile earned, keyed by the tile.
        vectors: Each tile's vector, keyed by weights, then instrument, then tile.
    """
    parquet.write_rows(
        [
            TileVector(tile, classes[tile], weights, instrument, tuple(vector.tolist()))
            for weights, by_instrument in vectors.items()
            for instrument, by_tile in by_instrument.items()
            for tile, vector in sorted(by_tile.items())
        ],
        RESULTS_SCHEMA,
        path,
    )


def read_tile_vectors(
    path: Path,
) -> tuple[list[str], list[str], dict[str, dict[str, np.ndarray]]]:
    """Return one evaluation's vectors, every model and instrument on one tile order.

    Args:
        path: The parquet file written by write_tile_vectors.

    Returns:
        tiles: Every tile any row holds, sorted.
        labels: The class of each, in the same order.
        vectors: Each tile's vector, NaN where it lacks the instrument, keyed by
            weights, then instrument. (T, D)
    """
    rows = parquet.read_rows(TileVector, RESULTS_SCHEMA, path)
    classes = {row.tile: row.label for row in rows}
    tiles = sorted(classes)
    at = {tile: index for index, tile in enumerate(tiles)}
    width = len(rows[0].vector)
    vectors = {}
    for row in rows:
        held = vectors.setdefault(row.weights, {}).setdefault(
            row.instrument, np.full((len(tiles), width), np.nan)
        )
        held[at[row.tile]] = row.vector
    return tiles, [classes[tile] for tile in tiles], vectors
