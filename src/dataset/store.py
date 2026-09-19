"""The published build of the dataset, and what its objects hold."""

from __future__ import annotations

import io
import json
import math
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
    MEASURED,
    META,
    NORTH,
)
from shared.disk import parquet
from torch.utils.data import DataLoader

from dataset.models.observation import Observation
from dataset.models.split import DatasetSplit

DISK_RESERVE_BYTES = 8 * 1024**3

TRAINING_SPLIT = "train"
VALIDATION_SPLIT = "validation"
SPLITS = (TRAINING_SPLIT, VALIDATION_SPLIT)


@dataclass(slots=True)
class DatasetBuild:
    """One published build of the dataset, read one object at a time.

    Attributes:
        root: Where the build sits on this machine.
        fetch: How one object is brought down when the root holds none of it.
        records: What every observation is, once the index is read, else None.
    """

    root: Path
    fetch: Callable[[str], bytes]
    records: list[ObservationMetadata] | None = None

    def read_object(self, path: str) -> bytes:
        """Return what one object of the build holds, off disk or fetched and kept.

        Args:
            path: Where it sits, relative to the build root, as the index names it.

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

    def read_observation_metadata_by_tile(
        self,
    ) -> dict[str, dict[str, list[ObservationMetadata]]]:
        """Return the observations of each tile, by the instrument that took them.

        Returns:
            standing: The rows of each sensor of each tile, keyed by identity.
        """
        standing = defaultdict(lambda: defaultdict(list))
        for one in self.read_observation_metadata():
            standing[one.tile][one.instrument].append(one)
        return {tile: dict(rows) for tile, rows in standing.items()}

    def read_row_by_instrument(self) -> dict[str, ObservationMetadata]:
        """Return one index row of each instrument the build reached.

        Returns:
            rows: One row per sensor, saying what each axis holds and how far it runs.
        """
        return {one.instrument: one for one in self.read_observation_metadata()}

    def read_ground_sample_by_instrument(self) -> dict[str, float]:
        """Return how much ground one sample of each instrument spans, over the build.

        Returns:
            sample_spacing_m: One length per sensor, its median finest ground axis.
        """
        standing = defaultdict(list)
        for one in self.read_observation_metadata():
            standing[one.instrument].append(min(one.sample_spacing_m))
        return {name: float(np.median(held)) for name, held in standing.items()}

    def read_heights(self, observations: Sequence[ObservationMetadata]) -> np.ndarray:
        """Return every height the elevation instrument measured over one tile.

        Args:
            observations: The tile's index rows of the elevation instrument.

        Returns:
            heights: One row per sample: north, east and height, in metres. (N, 3)

        Raises:
            ValueError: When none of them measured anything.
        """
        placed = []
        for record in observations:
            observation = self.read_observation(record.path)
            measured = observation.measured
            north, east = observation.distance_centre_m()
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

    def read_statistics_by_instrument(
        self, tiles: Collection[str] | None = None
    ) -> dict[str, dict[str, np.ndarray]]:
        """Return what each instrument's values run to, without reading one observation.

        Args:
            tiles: The tiles to pool over, or None for every one.

        Returns:
            statistics: Per sensor, the mean and deviation of its measurements.
        """
        standing: dict[str, list[tuple[np.ndarray, ...]]] = defaultdict(list)
        spreads: dict[str, list[int]] = {}
        for one in self.read_observation_metadata():
            if tiles is not None and one.tile not in tiles:
                continue
            # A spectral instrument is pooled a band at a time, every other whole.
            held = (
                (one.band_valid_count, one.band_mean, one.band_std)
                if WAVELENGTH in one.axes
                else (one.valid_count, one.value_mean, one.value_std)
            )
            # An observation measuring nothing leaves them unset, a sounder nan.
            if any(each is None for each in held):
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
        with np.load(io.BytesIO(self.read_object(path))) as held:
            # What the observation is, is stored beside its arrays as one json string.
            described = json.loads(str(held[META]))
            # Only these are read, so what is stored beside them stays packed.
            arrays = {
                name: held[name]
                for name in (described["measurement"], MEASURED, NORTH, EAST)
            }
        return Observation(
            instrument=described["instrument"],
            identifier=described["identifier"],
            measurement=described["measurement"],
            values=arrays[described["measurement"]],
            axes=tuple(described["axes"]),
            dims={name: tuple(held) for name, held in described["dims"].items()},
            measured=arrays[MEASURED],
            north=arrays[NORTH],
            east=arrays[EAST],
            described=described,
        )

    def loaders_by_split(
        self,
        sizes: Mapping[str, Mapping[str, int]],
        shapes: Mapping[str, tuple[int, ...]],
        collate: Callable,
        shares: Sequence[float],
        seed: int,
        elevation: str,
        batch_size: int,
        workers: int,
    ) -> dict[str, DataLoader]:
        """Return every split of the build in batches, whole tiles at a time.

        Args:
            sizes: How far a patch of each sensor runs along each axis it is cut on.
            shapes: The shape of one patch of each instrument as the model reads it.
            collate: How one batch of read tiles becomes what the model is handed.
            shares: The share of the observations each split holds, in the code's order.
            seed: What fixes which split a tile falls in.
            elevation: The instrument whose values give every surface patch its height.
            batch_size: How many tiles one step reads.
            workers: How many processes read tiles beside the training.

        Returns:
            loaders: One loader per split, the training one shuffled and the other not.
        """
        by_tile = self.read_observation_metadata_by_tile()
        wanted = dict(zip(SPLITS, shares, strict=True))
        counted = {
            tile: sum(len(held) for held in rows.values())
            for tile, rows in by_tile.items()
        }
        order = sorted(
            by_tile,
            key=lambda tile: (
                -counted[tile],
                random.Random(f"{seed}/{tile}").random(),
            ),
        )
        # A tile is one sample, so the splits are filled with whole tiles, the
        # heaviest first and each into the split standing furthest under its share.
        splits: dict[str, list[str]] = {name: [] for name in SPLITS}
        placed = dict.fromkeys(SPLITS, 0.0)
        for tile in order:
            name = min(
                SPLITS,
                key=lambda one: placed[one] / wanted[one] if wanted[one] else math.inf,
            )
            splits[name].append(tile)
            placed[name] += counted[tile]
        statistics = self.read_statistics_by_instrument(set(splits[TRAINING_SPLIT]))
        axes = {name: one.axes for name, one in self.read_row_by_instrument().items()}
        return {
            name: DataLoader(
                DatasetSplit(
                    self,
                    {tile: by_tile[tile] for tile in held},
                    axes,
                    statistics,
                    sizes,
                    shapes,
                    elevation,
                ),
                batch_size=batch_size,
                shuffle=name == TRAINING_SPLIT,  # Shuffle only for training
                num_workers=workers,
                persistent_workers=workers > 0,
                collate_fn=collate,
            )
            for name, held in splits.items()
        }
