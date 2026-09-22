"""The evaluation build's labels, read beside the tiles the dataset store reads."""

from __future__ import annotations

from analysis import paths as labelled
from analysis.ground_truth.artifacts import LABELS
from analysis.ground_truth.models.label import Label
from common.disk import parquet

from dataset.store import DatasetBuild


def read_label_by_tile(build: DatasetBuild) -> dict[str, str]:
    """Return the class the feature catalogue gave every tile the draw took.

    Args:
        build: The evaluation build, which carries its labels beside its index.

    Returns:
        classes: The class each tile earned, keyed by the tile it was drawn for.
    """
    held = build.read_table(labelled.LABELS_NAME, LABELS)
    labels = [parquet.build(Label, row) for row in held.to_pylist()]
    return {one.tile: one.label for one in labels}
