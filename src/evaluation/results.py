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
        instrument: The instrument the vector stands for.
        vector: Its tokens averaged, of unit length.
    """

    tile: str
    label: str
    instrument: str
    vector: tuple[float, ...]


RESULTS_SCHEMA = parquet.schema_of(TileVector)


def write_tile_vectors(
    path: Path,
    classes: Mapping[str, str],
    vectors: Mapping[str, Mapping[str, np.ndarray]],
) -> None:
    """Write one row per instrument and tile: its class and its vector.

    Args:
        path: The parquet file to write, whose directory is made if missing.
        classes: The class each tile earned, keyed by the tile.
        vectors: Each tile's vector, keyed by instrument, then tile.
    """
    parquet.write_rows(
        [
            TileVector(tile, classes[tile], instrument, tuple(vector.tolist()))
            for instrument, by_tile in vectors.items()
            for tile, vector in sorted(by_tile.items())
        ],
        RESULTS_SCHEMA,
        path,
    )


def read_tile_vectors(path: Path) -> tuple[list[str], list[str], dict[str, np.ndarray]]:
    """Return one evaluation's vectors, every instrument on one tile order.

    Args:
        path: The parquet file written by write_tile_vectors.

    Returns:
        tiles: Every tile any row holds, sorted.
        labels: The class of each, in the same order.
        vectors: Each tile's vector, NaN where it lacks the instrument, keyed by
            instrument. (T, D)
    """
    rows = parquet.read_rows(TileVector, RESULTS_SCHEMA, path)
    classes = {row.tile: row.label for row in rows}
    tiles = sorted(classes)
    at = {tile: index for index, tile in enumerate(tiles)}
    width = len(rows[0].vector)
    vectors = {}
    for row in rows:
        held = vectors.setdefault(row.instrument, np.full((len(tiles), width), np.nan))
        held[at[row.tile]] = row.vector
    return tiles, [classes[tile] for tile in tiles], vectors
