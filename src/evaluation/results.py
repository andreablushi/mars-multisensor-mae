"""What an evaluation keeps, one vector per tile and view, and what it comes to."""

from __future__ import annotations

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
        labels: The class of each tile.
        retrieval: Every tile's retrieval and ranking scores. (T,)
        knn: Accuracy and balanced accuracy of the kNN vote.
        classes: The classes, in the order both matrices hold them.
        confusion: Tiles of the row's class voted into the column's. (C, C)
        distances: The mean cosine distance between the tiles of two classes. (C, C)
        silhouette: How much nearer each tile sits to its class than to the next. (T,)
    """

    labels: list[str]
    retrieval: dict[str, np.ndarray]
    knn: dict[str, float]
    classes: list[str]
    confusion: np.ndarray
    distances: np.ndarray
    silhouette: np.ndarray


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


def write_tile_arrays(
    path: Path, by_instrument: Mapping[str, Mapping[str, np.ndarray]]
) -> None:
    """Write one tile's arrays, every one under <instrument>/<key>.

    Args:
        path: The npz file to write, whose directory is made if missing.
        by_instrument: Each instrument's arrays, keyed as
            reconstruction_and_attention names them.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        **{
            f"{name}/{key}": array
            for name, arrays in by_instrument.items()
            for key, array in arrays.items()
        },
    )
