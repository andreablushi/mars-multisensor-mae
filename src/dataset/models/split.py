"""One split of the dataset, each tile read as the patches of each instrument."""

from __future__ import annotations

import hashlib
import io
import json
import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from building.common.layout import Axis
from building.metadata.observation import ObservationMetadata
from torch.utils.data import Dataset

from dataset.models.patch import Patch
from dataset.patches import read_tile_patches
from logs.console import rich_logger

if TYPE_CHECKING:
    from dataset.store import DatasetBuild

log = rich_logger(__name__)

READY = "ready"


class DatasetSplit(Dataset):
    """Every tile of one split, each read as the patches of each instrument.

    Attributes:
        build: The build the tiles are read from.
        tiles: The index rows of each sensor of each tile, keyed by tile.
        identities: The tiles, in the order the split holds them.
        axes: What each axis of each instrument's values holds.
        statistics: What each sensor's values run to over the training split.
        sizes: How far a patch of each sensor runs along each axis it is cut on.
        pool: How many ground samples of a patch each instrument averages into one.
        shapes: The shape of one patch of each instrument as the model reads it.
        delay: The instrument whose rows give every surface patch its delay.
        ready: Where the tiles already read are kept, named for all that shapes them.
    """

    def __init__(
        self,
        build: DatasetBuild,
        tiles: Mapping[str, dict[str, list[ObservationMetadata]]],
        axes: Mapping[str, tuple[str, ...]],
        statistics: Mapping[str, dict[str, np.ndarray]],
        sizes: Mapping[str, Mapping[str, int]],
        pool: Mapping[str, int],
        shapes: Mapping[str, tuple[int, ...]],
        delay: str,
    ) -> None:
        """Keep what every read needs, and check every instrument can be normalised.

        Args:
            build: The build the tiles are read from.
            tiles: The index rows of each sensor of each tile, keyed by tile.
            axes: What each axis of each instrument's values holds.
            statistics: What each sensor's values run to over the training split.
            sizes: How far a patch of each sensor runs along each axis it is cut on.
            pool: How many ground samples of a patch each instrument averages into one.
            shapes: The shape of one patch of each instrument as the model reads it.
            delay: The instrument whose rows give every surface patch its delay.

        Raises:
            ValueError: When a sensor the model reads has no statistics to scale by.
        """
        self.build = build
        self.tiles = tiles
        self.identities = list(tiles)
        self.axes = axes
        self.statistics = statistics
        self.sizes = sizes
        self.pool = pool
        self.shapes = shapes
        self.delay = delay
        for name in sizes:
            if name not in statistics:
                raise ValueError(f"{name} has no finite statistics to normalise by")
        shaped = hashlib.sha256(
            json.dumps([sizes, pool, shapes, axes, delay], sort_keys=True).encode()
        )
        for name in sorted(sizes):
            shaped.update(statistics[name]["mean"].tobytes())
            shaped.update(statistics[name]["deviation"].tobytes())
        self.ready = f"{READY}/{shaped.hexdigest()[:16]}"

    def __len__(self) -> int:
        """Return how many reads the split holds.

        Returns:
            count: One per tile.
        """
        return len(self.identities)

    def __getitem__(self, index: int) -> tuple[dict[str, dict[str, np.ndarray]], str]:
        """Return what one read of a tile holds, and whose it is.

        Args:
            index: Which read of the split.

        Returns:
            sample: Each sensor's patches: values, valid and position.
            identity: The tile the read belongs to.

        Raises:
            ValueError: When the tile has no delay to stand on.
        """
        started = time.perf_counter()
        fetched = self.build.fetched_bytes
        identity = self.identities[index]
        kept = f"{self.ready}/{identity}.npz"
        stored = self.build.read_kept(kept)
        if stored is not None:
            sample = {}
            with np.load(io.BytesIO(stored)) as arrays:
                for packed in arrays.files:
                    name, key = packed.split("/")
                    sample.setdefault(name, {})[key] = arrays[packed]
            log.info(
                "tile %s read ready in %.2f s",
                identity,
                time.perf_counter() - started,
            )
            return sample, identity
        rows = self.tiles[identity]
        held = rows.get(self.delay)
        if not held:
            raise ValueError(f"{identity} has no {self.delay} to stand on")
        delays = self.build.read_delays(held)
        read = read_tile_patches(rows, self.build, self.sizes, self.pool, delays)
        sample = {
            name: patch_arrays(
                drawn, self.shapes[name], self.axes[name], self.statistics[name]
            )
            for name, drawn in read.items()
        }
        packed = io.BytesIO()
        np.savez(
            packed,
            **{
                f"{name}/{key}": array
                for name, arrays in sample.items()
                for key, array in arrays.items()
            },
        )
        self.build.keep(kept, packed.getvalue())
        log.info(
            "tile %s read in %.1f s, %.1f MB of it fetched",
            identity,
            time.perf_counter() - started,
            (self.build.fetched_bytes - fetched) / 1e6,
        )
        return sample, identity


def patch_arrays(
    patches: Sequence[Patch],
    shape: Sequence[int],
    axes: Sequence[str],
    statistics: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return one instrument's read patches as the arrays a model is handed.

    Args:
        patches: The patches, all of one instrument.
        shape: The shape of one patch of it as the model reads it.
        axes: What each axis of its values holds.
        statistics: What its values run to over the training split.

    Returns:
        arrays: The patches under "values", "valid" and "position".
    """
    valid_shape = tuple(
        held if holds in (Axis.GROUND, Axis.WAVELENGTH) else 1
        for held, holds in zip(shape, axes, strict=True)
    )
    scaled = [
        np.where(
            one.valid,
            (one.values.astype(np.float32) - statistics["mean"])
            / np.maximum(statistics["deviation"], 1e-6),
            0.0,
        )
        for one in patches
    ]
    return {
        "values": stacked(scaled, tuple(shape), np.float32),
        "valid": stacked([one.valid for one in patches], valid_shape, bool),
        "position": stacked(
            [
                np.array(
                    [
                        one.east_m,
                        one.north_m,
                        one.delay,
                        one.east_span_m,
                        one.north_span_m,
                        one.delay_span,
                    ],
                    np.float32,
                )
                for one in patches
            ],
            (6,),
            np.float32,
        ),
    }


def stacked(
    arrays: Sequence[np.ndarray], shape: tuple[int, ...], dtype: np.dtype | type
) -> np.ndarray:
    """Return same-shaped arrays as one array, empty where none were given.

    Args:
        arrays: The arrays, every one of that shape.
        shape: Their shape, which an empty list cannot say.
        dtype: What they hold, which an empty list cannot say either.

    Returns:
        stacked: The arrays along a new first axis. (K, *shape)
    """
    if not arrays:
        return np.zeros((0, *shape), dtype)
    return np.stack(arrays)
