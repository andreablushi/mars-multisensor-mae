"""The evaluation build: the class every tile carries, and the tiles themselves."""

from __future__ import annotations

import io
from collections.abc import Callable, Mapping

import numpy as np
import pyarrow.parquet as pq
from analysis import paths as labelled
from analysis.labels.artifacts import LABELS
from analysis.labels.models.label import Label
from torch.utils.data import DataLoader

from dataset.store import DatasetBuild, tile_loader


def read_label_by_tile(build: DatasetBuild) -> dict[str, str]:
    """Return the class the feature catalogue gave every tile the draw took.

    Args:
        build: The evaluation build, which carries its labels beside its index.

    Returns:
        classes: The class each tile earned, keyed by the tile it was drawn for.
    """
    held = pq.read_table(
        io.BytesIO(build.read_object(labelled.LABELS_NAME)), schema=LABELS
    )
    labels = [Label(**row) for row in held.to_pylist()]
    return {one.tile: one.label for one in labels}


def labelled_loader(
    build: DatasetBuild,
    classes: Mapping[str, str],
    statistics: Mapping[str, dict[str, np.ndarray]],
    sizes: Mapping[str, Mapping[str, int]],
    shapes: Mapping[str, tuple[int, ...]],
    collate: Callable,
    delay: str,
    batch_size: int,
    workers: int,
) -> DataLoader:
    """Return every labelled tile of the evaluation build in batches, in one order.

    Args:
        build: The evaluation build the tiles are read from.
        classes: The class each tile earned, which is what the build is read for.
        statistics: What each sensor's values are scaled by, which are the training
            build's so the model reads what it was fitted to.
        sizes: How far a patch of each sensor runs along each axis it is cut on.
        shapes: The shape of one patch of each instrument as the model reads it.
        collate: How one batch of read tiles becomes what the model is handed.
        delay: The instrument whose rows give every surface patch its delay.
        batch_size: How many tiles one step reads.
        workers: How many processes read tiles beside the measuring.

    Returns:
        loader: The labelled tiles, in batches, unshuffled so a run is repeatable.
    """
    by_tile = build.read_observation_metadata_by_tile()
    held = {tile: rows for tile, rows in by_tile.items() if tile in classes}
    return tile_loader(
        build,
        held,
        statistics,
        sizes,
        shapes,
        collate,
        delay,
        batch_size,
        workers,
        False,
    )
