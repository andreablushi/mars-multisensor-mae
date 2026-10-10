"""The results an evaluation keeps: tile vectors and reconstructed tiles."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from common.disk import parquet
from common.disk.files import atomic_path


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


@dataclass(frozen=True, slots=True)
class Reconstruction:
    """One instrument's patches of one tile, what the model was shown and wrote back.

    Attributes:
        values: The patches, zero where nothing was measured. (K, *P)
        measured: Whether each sample is a measurement. (K, *P')
        cell: The row and column of the grid cell each patch is drawn at. (K, 2)
        visible: Which patches the encoder read. (K,)
        hidden: Which patches the decoder wrote. (K,)
        umr: The patches written from the instrument's own tokens. (K, *P)
        cmr: The patches written from every other instrument's tokens. (K, *P)
    """

    values: np.ndarray
    measured: np.ndarray
    cell: np.ndarray
    visible: np.ndarray
    hidden: np.ndarray
    umr: np.ndarray
    cmr: np.ndarray


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


def write_reconstructions(
    path: Path, tiles: Mapping[str, Mapping[str, Reconstruction]]
) -> None:
    """Write every reconstructed tile into one compressed archive.

    Args:
        path: The archive to write, whose directory is made if missing.
        tiles: Each instrument's reconstruction, keyed by tile, then instrument.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as written, written.open("wb") as file:
        np.savez_compressed(
            file,
            **{
                f"{tile}/{name}/{field.name}": getattr(one, field.name)
                for tile, by_instrument in tiles.items()
                for name, one in by_instrument.items()
                for field in fields(Reconstruction)
            },
        )


def read_reconstructions(path: Path) -> dict[str, dict[str, Reconstruction]]:
    """Return every reconstructed tile one evaluation wrote.

    Args:
        path: The archive written by write_reconstructions.

    Returns:
        tiles: Each instrument's reconstruction, keyed by tile, then instrument.
    """
    held = defaultdict(lambda: defaultdict(dict))
    with np.load(path) as arrays:
        for packed in arrays.files:
            tile, name, field = packed.split("/")
            held[tile][name][field] = arrays[packed]
    return {
        tile: {name: Reconstruction(**one) for name, one in by_instrument.items()}
        for tile, by_instrument in held.items()
    }
