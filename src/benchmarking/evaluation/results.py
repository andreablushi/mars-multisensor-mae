"""What a benchmark writes: every test tile's consistency, by protocol and task."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from common.disk import parquet

from benchmarking.evaluation.metrics import DEPTHS
from configs.paths import predictions_path


@dataclass(frozen=True, slots=True)
class IcePrediction:
    """One test tile's consistency with ice at one depth, under one protocol and task.

    Attributes:
        tile: The tile.
        protocol: How its head was trained, "linear-probe" or "fine-tune".
        task: What its head tells, "classification" or "segmentation".
        depth: The depth, as SWIM's labels name it.
        consistency: The consistency, one for the tile or one per point, rows from
            the north and points from the west.
    """

    tile: str
    protocol: str
    task: str
    depth: str
    consistency: tuple[float, ...]


PREDICTIONS_SCHEMA = parquet.schema_of(IcePrediction)


def write_predictions(
    run_name: str,
    consistency: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
) -> Path:
    """Return where one run's benchmark predictions were written, replacing any.

    Args:
        run_name: What the run is called, which names its predictions.
        consistency: The consistency at each depth of every test tile, keyed by
            protocol and task, then by tile. (Z,) or (P * P, Z)

    Returns:
        path: The predictions.
    """
    path = predictions_path(run_name)
    parquet.write_rows(
        [
            IcePrediction(
                tile, protocol, task, depth, tuple(np.ravel(held[..., at]).tolist())
            )
            for (protocol, task), by_tile in consistency.items()
            for tile, held in sorted(by_tile.items())
            for at, depth in enumerate(DEPTHS)
        ],
        PREDICTIONS_SCHEMA,
        path,
    )
    return path
