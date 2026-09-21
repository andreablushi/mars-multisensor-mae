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
import pyarrow as pa
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
from common.disk import parquet
from torch.utils.data import DataLoader

from dataset.models.observation import Observation
from dataset.models.split import DatasetSplit

DISK_RESERVE_BYTES = 8 * 1024**3

# What MOLA stores beside its heights: the radargram row each of them sounds at.
DELAY_PLANE = "delay"

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

    def read_table(self, path: str, schema: pa.Schema | None = None) -> pa.Table:
        """Return one parquet object of the build, read out of the bytes it holds.

        Args:
            path: Where it sits, relative to the build root.
            schema: What to read it under, or None to take the file's own.

        Returns:
            table: The rows it holds.
        """
        return pq.read_table(io.BytesIO(self.read_object(path)), schema=schema)

    def read_observation_metadata(self) -> list[ObservationMetadata]:
        """Return what every observation of the build is, reading the index once.

        Returns:
            records: One row per observation, in the order the index holds them.
        """
        if self.records is None:
            held = self.read_table(built.OBSERVATION_METADATA_NAME)
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

    def read_axes_by_instrument(self) -> dict[str, tuple[str, ...]]:
        """Return what each axis of each instrument's values holds.

        Returns:
            axes: The axes of one sensor's values, in the array's own order, per sensor.
        """
        return {name: one.axes for name, one in self.read_row_by_instrument().items()}

    def read_ground_sample_by_instrument(self) -> dict[str, float]:
        """Return how much ground one sample of each instrument spans, over the build.

        Returns:
            sample_spacing_m: One length per sensor, its median finest ground axis.
        """
        standing = defaultdict(list)
        for one in self.read_observation_metadata():
            standing[one.instrument].append(min(one.sample_spacing_m))
        return {name: float(np.median(held)) for name, held in standing.items()}

    def read_delays(self, observations: Sequence[ObservationMetadata]) -> np.ndarray:
        """Return the row every sample of the delay instrument sounds at, over a tile.

        Args:
            observations: The tile's index rows of the instrument carrying the delay.

        Returns:
            delays: One row per sample: north, east and the delay row. (N, 3)

        Raises:
            ValueError: When none of them measured anything.
        """
        placed = []
        for record in observations:
            observation = self.read_observation(record.path, (DELAY_PLANE,))
            measured = observation.measured
            north, east = observation.distance_centre_m()
            rows = observation.beside[DELAY_PLANE]
            placed.append(
                np.stack([north[measured], east[measured], rows[measured]], axis=1)
            )  # (n, 3)
        delays = np.concatenate(placed).astype(np.float64)  # (N, 3)
        if not delays.size:
            raise ValueError(
                f"nothing measured over {[one.identity for one in observations]}"
            )
        return delays

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
            counts, mean, deviation = (
                np.asarray(each, dtype=np.float64) for each in held
            )
            # A band the observation never measured holds no mean and says so with
            # nan, which its count already tells apart and which would pool to nan.
            mean = np.where(counts > 0, mean, 0.0)
            deviation = np.where(counts > 0, deviation, 0.0)
            if not counts.sum() or not np.isfinite([mean, deviation]).all():
                continue
            standing[one.instrument].append((counts, mean, deviation))
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

    def read_observation(self, path: str, beside: Sequence[str] = ()) -> Observation:
        """Return one stored observation, read out of the object it was written as.

        Args:
            path: Where that object sits, as the index names it.
            beside: What else the instrument stores to read, none of it by default.

        Returns:
            observation: The observation, its arrays as the build wrote them.
        """
        with np.load(io.BytesIO(self.read_object(path))) as held:
            # What the observation is, is stored beside its arrays as one json string.
            described = json.loads(str(held[META]))
            # Only these are read, so what is not asked for stays packed.
            arrays = {
                name: held[name]
                for name in (described["measurement"], MEASURED, NORTH, EAST, *beside)
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
            beside={name: arrays[name] for name in beside},
            described=described,
        )

    def read_training_statistics(
        self, shares: Sequence[float], seed: int
    ) -> dict[str, dict[str, np.ndarray]]:
        """Return what each instrument's values run to over the training split alone.

        Args:
            shares: The share of the observations each split holds, in the code's order.
            seed: What fixes which split a tile falls in.

        Returns:
            statistics: Per sensor, the mean and deviation of its measurements.
        """
        splits = split_tiles(self.read_observation_metadata_by_tile(), shares, seed)
        return self.read_statistics_by_instrument(set(splits[TRAINING_SPLIT]))

    def loaders_by_split(
        self,
        sizes: Mapping[str, Mapping[str, int]],
        shapes: Mapping[str, tuple[int, ...]],
        collate: Callable,
        shares: Sequence[float],
        seed: int,
        delay: str,
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
            delay: The instrument whose rows give every surface patch its delay.
            batch_size: How many tiles one step reads.
            workers: How many processes read tiles beside the training.

        Returns:
            loaders: One loader per split, the training one shuffled and the other not.
        """
        by_tile = self.read_observation_metadata_by_tile()
        splits = split_tiles(by_tile, shares, seed)
        statistics = self.read_statistics_by_instrument(set(splits[TRAINING_SPLIT]))
        axes = self.read_axes_by_instrument()
        return {
            name: tile_loader(
                self,
                {tile: by_tile[tile] for tile in held},
                axes,
                statistics,
                sizes,
                shapes,
                collate,
                delay,
                batch_size,
                workers,
                shuffle=name == TRAINING_SPLIT,
            )
            for name, held in splits.items()
        }


def split_tiles(
    by_tile: Mapping[str, Mapping[str, list[ObservationMetadata]]],
    shares: Sequence[float],
    seed: int,
) -> dict[str, list[str]]:
    """Return which tiles each split of a build holds.

    Args:
        by_tile: The rows of each sensor of each tile, keyed by identity.
        shares: The share of the observations each split holds, in the code's order.
        seed: What fixes which split a tile falls in.

    Returns:
        splits: The tiles of each split, keyed as the code names the splits.
    """
    wanted = dict(zip(SPLITS, shares, strict=True))
    counted = {
        tile: sum(len(held) for held in rows.values()) for tile, rows in by_tile.items()
    }
    order = sorted(
        by_tile,
        key=lambda tile: (-counted[tile], random.Random(f"{seed}/{tile}").random()),
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
    return splits


def tile_loader(
    build: DatasetBuild,
    tiles: Mapping[str, dict[str, list[ObservationMetadata]]],
    axes: Mapping[str, tuple[str, ...]],
    statistics: Mapping[str, dict[str, np.ndarray]],
    sizes: Mapping[str, Mapping[str, int]],
    shapes: Mapping[str, tuple[int, ...]],
    collate: Callable,
    delay: str,
    batch_size: int,
    workers: int,
    *,
    shuffle: bool = False,
) -> DataLoader:
    """Return one set of tiles of a build in batches, each tile read whole.

    Args:
        build: The build the tiles are read from.
        tiles: The index rows of each sensor of each tile, keyed by tile.
        axes: What each axis of each instrument's values holds, which is the model's
            own reading of them and not whatever build the tiles came from.
        statistics: What each sensor's values are scaled by, whatever they were
            pooled over.
        sizes: How far a patch of each sensor runs along each axis it is cut on.
        shapes: The shape of one patch of each instrument as the model reads it.
        collate: How one batch of read tiles becomes what the model is handed.
        delay: The instrument whose rows give every surface patch its delay.
        batch_size: How many tiles one step reads.
        workers: How many processes read tiles beside the work.
        shuffle: Whether the tiles are read in a new order every pass.

    Returns:
        loader: The tiles, in batches.
    """
    return DataLoader(
        DatasetSplit(build, tiles, axes, statistics, sizes, shapes, delay),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=workers > 0,
        collate_fn=collate,
    )
