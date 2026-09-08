"""The index of a build, read into the samples a run draws its patches from."""

from __future__ import annotations

import io
import json
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence

import pyarrow.parquet as pq
from building import paths as built
from building.metadata.feature import FeatureMetadata
from building.metadata.observation import ObservationMetadata
from shared.disk import parquet
from shared.models.feature import Feature

from dataset.models.sample import Sample
from dataset.models.settings import Settings
from dataset.seed import seeded_number
from dataset.store.artifact import Build


def read_manifest(build: Build) -> dict:
    """Return what the build says it is, as it wrote it.

    Args:
        build: The published build to read.

    Returns:
        manifest: When it was built, from what, and which instruments it reached.
            A build that published no format version carries none here either.
    """
    return json.loads(build.read(built.DATASET_MANIFEST_NAME))


def read_observations(build: Build) -> list[ObservationMetadata]:
    """Return what every crop of the build is, without reading one of them.

    Args:
        build: The published build to read.

    Returns:
        records: One row per crop, in the order the index holds them.
    """
    return _rows(build, built.OBSERVATION_METADATA_NAME, ObservationMetadata)


def read_features(build: Build) -> dict[tuple[str, str], FeatureMetadata]:
    """Return every feature the build considered, keyed by what tells it apart.

    Args:
        build: The published build to read.

    Returns:
        features: One row per feature, keyed by its class and its name.
    """
    rows = _rows(build, built.FEATURE_METADATA_NAME, FeatureMetadata)
    return {one.identity: one for one in rows}


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


def drawn_sample(
    frame: Feature,
    observations: Sequence[ObservationMetadata],
    settings: Settings,
    epoch: int = 0,
) -> Sample:
    """Return one feature and a draw of the crops the build holds of it.

    Args:
        frame: The feature every one of them was cropped to.
        observations: Its own index rows, which the draw is made from.
        settings: The settled choices, which say how many of each instrument a
            sample draws and what number fixes that draw.
        epoch: Which pass over the dataset this is, so one feature draws
            differently from one pass to the next and the same within one.

    Returns:
        sample: The feature and what was drawn of it, at most as many of each
            instrument as the config asks for, the instruments in name order.
    """
    standing: dict[str, list[ObservationMetadata]] = defaultdict(list)
    for one in observations:
        standing[one.instrument].append(one)
    identity = (frame.feature_class, frame.feature_name)
    drawing = random.Random(seeded_number("/".join(identity), settings.seed + epoch))
    drawn = [
        one
        for instrument in sorted(standing)
        for one in drawing.sample(
            standing[instrument],
            min(settings.per_instrument, len(standing[instrument])),
        )
    ]
    return Sample(feature=frame, observations=tuple(drawn))


def _rows[Row](build: Build, name: str, model: type[Row]) -> list[Row]:
    """Return one parquet file of the index, built back into its own rows.

    Args:
        build: The published build to read it off.
        name: What the file is called at the build's own root.
        model: The row model to build each row into.

    Returns:
        rows: The rows, in the order the file holds them. The file is read under
            no schema, so a build carrying columns the model does not declare is
            read all the same.
    """
    held = pq.read_table(io.BytesIO(build.read(name)))
    return [parquet.build(model, row) for row in held.to_pylist()]
