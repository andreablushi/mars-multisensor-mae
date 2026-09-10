"""The published build of the dataset, and what its objects hold."""

from __future__ import annotations

import io
import json
import math
import random
from collections import defaultdict
from collections.abc import Callable, Collection
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from building import paths as built
from building.common.layout import WAVELENGTH
from building.metadata.feature import FeatureMetadata
from building.metadata.observation import ObservationMetadata
from building.preprocessing.common.store import (
    EAST,
    INSIDE,
    MEASURED,
    META,
    NORTH,
    VALID,
)
from shared.disk import parquet
from shared.disk.files import atomic_path

from dataset.models.observation import Observation


@dataclass(slots=True)
class DatasetBuild:
    """One published build of the dataset, read one object at a time.

    Attributes:
        root: Where the build sits on this machine. A build brought down whole
            fills it, and a build read from elsewhere fills it as it is read,
            so a later pass over the same observations fetches none of them.
        fetch: How one object is brought down when the root holds none of it,
            and None for a build that is already whole on disk.
    """

    root: Path
    fetch: Callable[[str], bytes] | None = None

    def read_object(self, path: str) -> bytes:
        """Return what one object of the build holds, fetching it only once.

        Args:
            path: Where it sits, relative to the build's own root, as the index
                names it.

        Returns:
            data: The bytes of that object.

        Raises:
            FileNotFoundError: When the root holds no such object and the build
                was given nowhere to fetch it from.
        """
        held = self.root / path
        # What one pass fetched every later pass reads off the run's own disk.
        if held.is_file():
            return held.read_bytes()
        if self.fetch is None:
            raise FileNotFoundError(held)
        data = self.fetch(path)
        # Staged under a name of its own, so a killed run leaves no half object.
        with atomic_path(held) as staged:
            staged.write_bytes(data)
        return data

    def read_manifest(self) -> dict:
        """Return what the build says it is, as it wrote it.

        Returns:
            manifest: When it was built, from what, and which instruments it
                reached.
        """
        return json.loads(self.read_object(built.DATASET_MANIFEST_NAME))

    def read_observation_metadata(self) -> list[ObservationMetadata]:
        """Return what every observation of the build is, without reading one.

        Returns:
            records: One row per observation, in the order the index holds them.
                The file is read under no schema, so a build carrying columns
                the row model does not declare is read all the same.
        """
        held = pq.read_table(
            io.BytesIO(self.read_object(built.OBSERVATION_METADATA_NAME))
        )
        return [parquet.build(ObservationMetadata, row) for row in held.to_pylist()]

    def read_feature_metadata(self) -> dict[tuple[str, str], FeatureMetadata]:
        """Return what every feature of the build is, keyed by what tells it apart.

        Returns:
            features: Each feature's own row, carrying the frame its
                observations are placed against, which is the one place the
                build writes an absolute position down.
        """
        held = pq.read_table(io.BytesIO(self.read_object(built.FEATURE_METADATA_NAME)))
        return {
            one.identity: replace(
                one, centre_lon=one.frame.centre_lon, centre_lat=one.frame.centre_lat
            )
            for one in (parquet.build(FeatureMetadata, row) for row in held.to_pylist())
        }

    def read_observation_metadata_by_feature(
        self,
    ) -> dict[tuple[str, str], dict[str, list[ObservationMetadata]]]:
        """Return the observations of each feature, by the instrument that took them.

        Returns:
            standing: The rows of each instrument of each feature, keyed by
                what tells the feature apart and then as ODE names the
                instrument, in the order the index holds them.
        """
        standing = defaultdict(lambda: defaultdict(list))
        for one in self.read_observation_metadata():
            standing[one.feature][one.instrument].append(one)
        return {feature: dict(rows) for feature, rows in standing.items()}

    def split_features(
        self, shares: dict[str, float], seed: int
    ) -> dict[str, list[tuple[str, str]]]:
        """Return which features each split holds, whole features at a time.

        Args:
            shares: How much of the build each split holds, keyed by its name.
            seed: The number that fixes where a feature falls.

        Returns:
            splits: The features of each split, keyed as the shares name them. A
                feature falls in one split by its name alone, so no two patches
                of it straddle two splits and a later build that adds features
                leaves the ones already placed where they were.
        """
        total = sum(shares.values())
        splits: dict[str, list[tuple[str, str]]] = {name: [] for name in shares}
        for identity in self.read_observation_metadata_by_feature():
            drawn = random.Random(f"{seed}/{'/'.join(identity)}").random() * total
            running = 0.0
            for name in shares:
                running += shares[name]
                if drawn < running:
                    break
            splits[name].append(identity)
        return splits

    def compute_stats(
        self, features: Collection[tuple[str, str]] | None = None
    ) -> dict[str, dict[str, float]]:
        """Return what each instrument's values run to, without reading one observation.

        Args:
            features: The features to pool over, which a training run keeps to
                its own split so no test feature is seen, or None for every one.

        Returns:
            statistics: One entry per instrument whose values are the same
                values from one observation to the next, keyed as ODE names it,
                holding how many were pooled, their mean, and their deviation. A
                spectral instrument is absent: the masking drops the bands each
                of its observations refuses, so band three of one is not band
                three of another, and a patch of one is normalised by the band
                moments its own index row carries.
        """
        standing: dict[str, list[ObservationMetadata]] = defaultdict(list)
        for one in self.read_observation_metadata():
            # An observation measuring nothing leaves them unset, a sounder nan.
            moments = (one.value_mean, one.value_std)
            if (
                (features is None or one.feature in features)
                and WAVELENGTH not in one.axes
                and one.valid_count
                and all(held is not None and math.isfinite(held) for held in moments)
            ):
                standing[one.instrument].append(one)
        statistics = {}
        for instrument, held in standing.items():
            total = sum(one.valid_count for one in held)
            mean = sum(one.valid_count * one.value_mean for one in held) / total
            second = (
                sum(
                    one.valid_count * (one.value_std**2 + one.value_mean**2)
                    for one in held
                )
                / total
            )
            statistics[instrument] = {
                "count": total,
                "mean": mean,
                "deviation": math.sqrt(max(second - mean**2, 0.0)),
            }
        return statistics

    def read_observation(self, path: str) -> Observation:
        """Return one stored observation, read out of the object it was written as.

        Args:
            path: Where that object sits, as the index names it.

        Returns:
            observation: The observation, its arrays as the build wrote them.
        """
        with np.load(io.BytesIO(self.read_object(path))) as held:
            # What the observation is, is stored beside its arrays as one json string.
            described = json.loads(str(held[META]))
            arrays = {name: held[name] for name in held.files if name != META}
        for name in (INSIDE, VALID):
            arrays.pop(name, None)
        return Observation(
            instrument=described["instrument"],
            identifier=described["identifier"],
            measurement=described["measurement"],
            values=arrays.pop(described["measurement"]),
            axes=tuple(described["axes"]),
            dims={name: tuple(held) for name, held in described["dims"].items()},
            measured=arrays.pop(MEASURED),
            north=arrays.pop(NORTH),
            east=arrays.pop(EAST),
            # What the pops left is what the instrument stores beside its values.
            beside=arrays,
        )
