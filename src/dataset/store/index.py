"""The index of a build, read into the observations a run draws its patches from."""

from __future__ import annotations

import io
import json
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence

import pyarrow.parquet as pq
from building import paths as built
from building.metadata.observation import ObservationMetadata
from shared.disk import parquet

from dataset.store.read_dh_volumes import Build


def read_manifest(build: Build) -> dict:
    """Return what the build says it is, as it wrote it.

    Args:
        build: The published build to read.

    Returns:
        manifest: When it was built, from what, and which instruments it reached.
    """
    return json.loads(build.read_object(built.DATASET_MANIFEST_NAME))


def read_observations(build: Build) -> list[ObservationMetadata]:
    """Return what every crop of the build is, without reading one of them.

    Args:
        build: The published build to read.

    Returns:
        records: One row per crop, in the order the index holds them. The file
            is read under no schema, so a build carrying columns the row model
            does not declare is read all the same.
    """
    held = pq.read_table(io.BytesIO(build.read_object(built.OBSERVATION_METADATA_NAME)))
    return [parquet.build(ObservationMetadata, row) for row in held.to_pylist()]


def observations_by_feature(
    observations: Iterable[ObservationMetadata],
) -> dict[tuple[str, str], list[ObservationMetadata]]:
    """Return the crops of each feature, keyed by what tells that feature apart.

    Args:
        observations: The index rows to group, in any order.

    Returns:
        standing: The rows of each feature, in the order the index held them.
    """
    standing: dict[tuple[str, str], list[ObservationMetadata]] = defaultdict(list)
    for one in observations:
        standing[one.feature].append(one)
    return dict(standing)


def drawn_observations(
    observations: Sequence[ObservationMetadata], config: dict
) -> list[ObservationMetadata]:
    """Return a draw of the crops the build holds of one feature.

    Args:
        observations: One feature's own index rows, which the draw is made from.
        config: The choices a read is made with, which say how many of each
            instrument a sample draws and what number fixes that draw.

    Returns:
        drawn: At most as many of each instrument as the config asks for, the
            instruments in name order. The draw is fixed by the feature's own
            name, so every process draws the same crops of it.
    """
    standing: dict[str, list[ObservationMetadata]] = defaultdict(list)
    for one in observations:
        standing[one.instrument].append(one)
    drawing = random.Random(f"{config['seed']}/{'/'.join(observations[0].feature)}")
    return [
        one
        for instrument in sorted(standing)
        for one in drawing.sample(
            standing[instrument],
            min(config["per_instrument"], len(standing[instrument])),
        )
    ]
