"""What an evaluation keeps, one vector per tile and view, and what it comes to."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from common.disk import parquet


@dataclass(frozen=True, slots=True)
class TileVector:
    """One labelled tile, its class and its vector over one pooling and view.

    Attributes:
        tile: The tile.
        label: The class it earned.
        pooling: How the tokens were pooled, CLS or GAP.
        instrument: The instrument the vector stands for, or all of them.
        vector: Its tokens pooled, of unit length.
    """

    tile: str
    label: str
    pooling: str
    instrument: str
    vector: tuple[float, ...]


RESULTS_SCHEMA = parquet.schema_of(TileVector)


@dataclass(frozen=True, slots=True)
class Metrics:
    """Everything one view of the tiles comes to.

    Attributes:
        tiles: The tiles.
        labels: The class of each, in the same order.
        retrieval: Every tile's retrieval and ranking scores. (T,)
        knn: Accuracy and balanced accuracy of the kNN vote.
        classes: The classes, in the order both matrices hold them.
        confusion: Tiles of the row's class voted into the column's. (C, C)
        distances: The mean cosine distance between the tiles of two classes. (C, C)
        projection: Where each tile lands on the UMAP plane. (T, 2)
    """

    tiles: list[str]
    labels: list[str]
    retrieval: dict[str, np.ndarray]
    knn: dict[str, float]
    classes: list[str]
    confusion: np.ndarray
    distances: np.ndarray
    projection: np.ndarray


def write_tile_vectors(
    path: Path,
    classes: Mapping[str, str],
    vectors: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
) -> None:
    """Write one row per pooling, view and tile: its class and its vector.

    Args:
        path: The parquet file to write, whose directory is made if missing.
        classes: The class each tile earned, keyed by the tile.
        vectors: Each tile's vector, keyed by pooling, then view, then tile.
    """
    parquet.write_rows(
        [
            TileVector(tile, classes[tile], pooling, view, tuple(vector.tolist()))
            for pooling, by_view in vectors.items()
            for view, by_tile in by_view.items()
            for tile, vector in sorted(by_tile.items())
        ],
        RESULTS_SCHEMA,
        path,
    )


def read_tile_vectors(
    path: Path,
) -> tuple[list[str], list[str], dict[tuple[str, str], np.ndarray]]:
    """Return one evaluation's vectors, every pooling and view on one tile order.

    Args:
        path: The parquet file written by write_tile_vectors.

    Returns:
        tiles: Every tile, sorted.
        labels: The class of each, in the same order.
        vectors: Each tile's vector, keyed by pooling and view. (T, D)
    """
    rows = parquet.read_rows(TileVector, RESULTS_SCHEMA, path)
    classes = {row.tile: row.label for row in rows}
    tiles = sorted(classes)
    by_view = defaultdict(dict)
    for row in rows:
        by_view[row.pooling, row.instrument][row.tile] = row.vector
    vectors = {
        view: np.array([held[tile] for tile in tiles]) for view, held in by_view.items()
    }
    return tiles, [classes[tile] for tile in tiles], vectors


def write_reconstruction(
    path: Path, tile: str, reconstruction: Mapping[str, Mapping[str, np.ndarray]]
) -> None:
    """Write one tile's reconstruction, every array under <instrument>/<key>.

    Args:
        path: The npz file to write, whose directory is made if missing.
        tile: The tile rebuilt.
        reconstruction: Each instrument's arrays, keyed as reconstruct names them.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        tile=np.array(tile),
        **{
            f"{name}/{key}": array
            for name, arrays in reconstruction.items()
            for key, array in arrays.items()
        },
    )


def read_reconstruction(path: Path) -> tuple[str, dict[str, dict[str, np.ndarray]]]:
    """Return one tile's reconstruction, as write_reconstruction wrote it.

    Args:
        path: The npz file.

    Returns:
        tile: The tile rebuilt.
        reconstruction: Each instrument's arrays, keyed as reconstruct names them.
    """
    reconstruction = defaultdict(dict)
    with np.load(path) as arrays:
        for packed in arrays.files:
            if packed != "tile":
                name, key = packed.split("/")
                reconstruction[name][key] = arrays[packed]
        return str(arrays["tile"]), dict(reconstruction)
