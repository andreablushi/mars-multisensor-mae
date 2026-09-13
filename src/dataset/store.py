"""The published build of the dataset, and what its objects hold."""

from __future__ import annotations

import io
import json
import os
import random
import shutil
from collections import defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from building import paths as built
from building.common.layout import WAVELENGTH
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
from torch.utils.data import DataLoader

from config.schema import Config
from dataset.models.observation import Observation
from dataset.models.split import DatasetSplit
from dataset.patches import PATCHES_PER_STEP, patch_lengths

# What is left free on the disk a run is given, under which nothing more is kept.
DISK_RESERVE_BYTES = 8 * 1024**3

# The splits a build is read in, which the config gives a share of the features each.
TRAINING_SPLIT = "train"
VALIDATION_SPLIT = "validation"
TEST_SPLIT = "test"
SPLITS = (TRAINING_SPLIT, VALIDATION_SPLIT, TEST_SPLIT)


@dataclass(slots=True)
class DatasetBuild:
    """One published build of the dataset, read one object at a time.

    Attributes:
        root: Where the build sits on this machine.
        fetch: How one object is brought down when the root holds none of it.
        records: What every observation of the build is, once the index has
            been read, and None until then.
    """

    root: Path
    fetch: Callable[[str], bytes]
    records: list[ObservationMetadata] | None = None

    def read_object(self, path: str) -> bytes:
        """Return what one object of the build holds, off disk or from the store.

        A build runs to some hundred gigabytes and the disk a run is given holds
        a fraction of it, but a run reads the same few thousand objects of it
        once an epoch. What is fetched is kept while there is room, so an epoch
        after the first reads off disk instead of over the network.

        Args:
            path: Where it sits, relative to the build's own root, as the index
                names it.

        Returns:
            data: The bytes of that object.
        """
        held = self.root / path
        if held.is_file():
            return held.read_bytes()
        data = self.fetch(path)
        held.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(held.parent).free - len(data) > DISK_RESERVE_BYTES:
            # Written whole then moved, so a reader beside this one finds it finished.
            temporary = held.with_suffix(f"{held.suffix}.{os.getpid()}")
            temporary.write_bytes(data)
            temporary.replace(held)
        return data

    def read_observation_metadata(self) -> list[ObservationMetadata]:
        """Return what every observation of the build is, reading the index once.

        Returns:
            records: One row per observation, in the order the index holds them.
        """
        if self.records is None:
            held = pq.read_table(
                io.BytesIO(self.read_object(built.OBSERVATION_METADATA_NAME))
            )
            self.records = [
                parquet.build(ObservationMetadata, row) for row in held.to_pylist()
            ]
        return self.records

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

    def read_row_by_instrument(self) -> dict[str, ObservationMetadata]:
        """Return one index row of each instrument the build reached.

        Returns:
            rows: One row per instrument, keyed as ODE names it, which says
                what each axis of its values holds and how far each runs.
        """
        return {one.instrument: one for one in self.read_observation_metadata()}

    def read_ground_sample_by_instrument(self) -> dict[str, float]:
        """Return how much ground one sample of each instrument spans, over the build.

        Returns:
            ground_sample_m: One length per instrument, keyed as ODE names it:
                the median over its observations of the finest of its ground
                axes, so one scan of an odd resolution does not settle it.
        """
        standing = defaultdict(list)
        for one in self.read_observation_metadata():
            standing[one.instrument].append(min(one.ground_sample_m))
        return {name: float(np.median(held)) for name, held in standing.items()}

    def read_heights(self, observations: Sequence[ObservationMetadata]) -> np.ndarray:
        """Return every height the elevation instrument measured over one feature.

        Args:
            observations: The feature's index rows of the elevation instrument.

        Returns:
            heights: One row per measured sample, pooled over the observations:
                how far north of the feature centre it sits, how far east, and
                how high above the areoid the ground stands there, in metres.
                (N, 3)

        Raises:
            ValueError: When none of them measured anything.
        """
        placed = []
        for record in observations:
            observation = self.read_observation(record.path)
            measured = observation.measured
            north, east = observation.ground_metres()
            placed.append(
                np.stack(
                    [north[measured], east[measured], observation.values[measured]],
                    axis=1,
                )
            )  # (n, 3)
        heights = np.concatenate(placed).astype(np.float64)  # (N, 3)
        if not heights.size:
            raise ValueError(
                f"nothing measured over {[one.identity for one in observations]}"
            )
        return heights

    def compute_stats(
        self, features: Collection[tuple[str, str]] | None = None
    ) -> dict[str, dict[str, np.ndarray]]:
        """Return what each instrument's values run to, without reading one observation.

        Args:
            features: The features to pool over, or None for every one.

        Returns:
            statistics: One entry per instrument, keyed as ODE names it, holding
                how many measurements it pooled, their mean and their deviation.
                A spectral instrument is pooled a band at a time and its moments
                run along its own wavelength axis, so they divide one of its
                patches as a single number divides any other.
        """
        standing: dict[str, list[tuple[np.ndarray, ...]]] = defaultdict(list)
        spreads: dict[str, list[int]] = {}
        for one in self.read_observation_metadata():
            # A spectral instrument is pooled a band at a time, every other whole.
            banded = WAVELENGTH in one.axes
            held = (
                (one.band_valid_count, one.band_mean, one.band_std)
                if banded
                else (one.valid_count, one.value_mean, one.value_std)
            )
            # An observation measuring nothing leaves them unset, a sounder nan.
            if (features is not None and one.feature not in features) or any(
                each is None for each in held
            ):
                continue
            moments = tuple(np.asarray(each, dtype=np.float64) for each in held)
            if not moments[0].sum() or not np.isfinite(moments[1:]).all():
                continue
            standing[one.instrument].append(moments)
            spreads[one.instrument] = [
                -1 if holds == WAVELENGTH else 1 for holds in one.axes
            ]
        statistics = {}
        for instrument, held in standing.items():
            counts = np.sum([one[0] for one in held], axis=0)
            total = np.maximum(counts, 1.0)
            mean = np.sum([one[0] * one[1] for one in held], axis=0) / total
            second = (
                np.sum([one[0] * (one[2] ** 2 + one[1] ** 2) for one in held], axis=0)
                / total
            )
            deviation = np.sqrt(np.maximum(second - mean**2, 0.0))
            # A band nothing ever measured leaves a patch of it where it stands.
            spread = spreads[instrument] if counts.shape else ()
            statistics[instrument] = {
                "count": counts,
                "mean": np.where(counts > 0, mean, 0.0)
                .reshape(spread)
                .astype(np.float32),
                "deviation": np.where(counts > 0, deviation, 1.0)
                .reshape(spread)
                .astype(np.float32),
            }
        return statistics

    def read_observation(self, path: str) -> Observation:
        """Return one stored observation, read out of the object it was written as.

        Args:
            path: Where that object sits, as the index names it.

        Returns:
            observation: The observation, its arrays as the build wrote them.
        """
        # What INSIDE and VALID hold is what MEASURED holds, and reading one of
        # them costs as much as reading the values, so they are left unread.
        skipped = (META, INSIDE, VALID)
        with np.load(io.BytesIO(self.read_object(path))) as held:
            # What the observation is, is stored beside its arrays as one json string.
            described = json.loads(str(held[META]))
            arrays = {name: held[name] for name in held.files if name not in skipped}
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
            described=described,
        )

    def read_patch_layout(
        self, sizes: Mapping[str, int]
    ) -> tuple[dict[str, tuple[int, ...]], dict[str, float]]:
        """Return the shape of one patch of each instrument, and how far two sit apart.

        Args:
            sizes: How far a patch of each instrument runs along an axis it is
                cut on, keyed as ODE names it.

        Returns:
            shapes: The shape of one patch of each instrument, keyed as ODE
                names it.
            strides: How far apart two neighbouring patch centres of each
                instrument sit, in metres, which sets the shortest period its
                positions are read at.
        """
        rows = self.read_row_by_instrument()
        ground = self.read_ground_sample_by_instrument()
        return (
            {
                name: patch_lengths(rows[name].shape, rows[name].axes, size)
                for name, size in sizes.items()
            },
            {name: size * ground[name] for name, size in sizes.items()},
        )

    def loaders_by_split(
        self,
        config: Config,
        sizes: Mapping[str, int],
        shapes: Mapping[str, tuple[int, ...]],
        collate: Callable,
    ) -> dict[str, DataLoader]:
        """Return every split of the build in batches, whole features at a time.

        A feature falls in one split by its name alone, so no two patches of it
        straddle two splits and a later build that adds features leaves the
        ones already placed where they were.

        A feature runs to more patches than one step can carry, so a read draws
        what a step holds beside the other features of its batch. The training
        split draws anew every read, so a run reads every patch of a feature
        over its epochs; the others draw against the seed, so what one pass
        measured the next measures again.

        Args:
            config: What the run reads and how much of it one step reads: the
                share of the build each split holds, the number that fixes
                where a feature falls, the instrument every surface patch
                takes its height from, and how many features and processes a
                step runs, which is also what settles how many patches of each
                instrument one read draws.
            sizes: How far a patch of each instrument runs along an axis it is
                cut on, keyed as ODE names it, which is also which instruments
                the model reads.
            shapes: The shape of one patch of each instrument as the model
                reads it.
            collate: How one batch of drawn features becomes what the model is
                handed.

        Returns:
            loaders: One loader per split, keyed as `SPLITS` names it, the
                training one shuffled and every other read in the index's own
                order.
        """
        by_feature = self.read_observation_metadata_by_feature()
        shares = config.dataset.split
        seed = config.dataset.seed
        total = sum(shares)
        splits: dict[str, list[tuple[str, str]]] = {name: [] for name in SPLITS}
        for identity in by_feature:
            drawn = random.Random(f"{seed}/{'/'.join(identity)}").random() * total
            running = 0.0
            for name, share in zip(SPLITS, shares, strict=True):
                running += share
                if drawn < running:
                    break
            splits[name].append(identity)
        statistics = self.compute_stats(set(splits[TRAINING_SPLIT]))
        axes = {name: one.axes for name, one in self.read_row_by_instrument().items()}
        budget = max(PATCHES_PER_STEP // config.training.batch_size, 1)
        return {
            name: DataLoader(
                DatasetSplit(
                    self,
                    {identity: by_feature[identity] for identity in held},
                    axes,
                    statistics,
                    sizes,
                    shapes,
                    config.model.elevation,
                    budget,
                    None if name == TRAINING_SPLIT else seed,
                ),
                batch_size=config.training.batch_size,
                shuffle=name == TRAINING_SPLIT,
                num_workers=config.training.workers,
                persistent_workers=config.training.workers > 0,
                collate_fn=collate,
            )
            for name, held in splits.items()
        }
