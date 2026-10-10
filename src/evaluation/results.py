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


def read_tile_vectors(path: Path) -> list[TileVector]:
    """Return every row one evaluation wrote.

    Args:
        path: The parquet file written by write_tile_vectors.

    Returns:
        rows: One per model, instrument and tile.
    """
    return parquet.read_rows(TileVector, RESULTS_SCHEMA, path)
