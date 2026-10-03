"""The splits of a build, and the loaders reading their tiles in batches."""

from __future__ import annotations

import gc
import random
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from building.common.layout import Axis
from building.metadata.observation import ObservationMetadata
from common.disk.files import atomic_path
from torch.utils.data import DataLoader, Dataset

from configs.paths import ready_tile_path
from dataset.models.patch import Patch
from dataset.patches import read_surface_delays, read_tile_patches
from dataset.store import DatasetBuild

TRAINING_SPLIT = "train"
VALIDATION_SPLIT = "validation"
SPLITS = (TRAINING_SPLIT, VALIDATION_SPLIT)


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
            key=lambda one: placed[one] / wanted[one],
        )
        splits[name].append(tile)
        placed[name] += counted[tile]
    return splits


def loaders_by_split(
    build: DatasetBuild,
    sizes: Mapping[str, Mapping[str, int]],
    pool: Mapping[str, int],
    shapes: Mapping[str, tuple[int, ...]],
    collate: Callable,
    shares: Sequence[float],
    seed: int,
    delay: str,
    batch_size: int,
    workers: int,
) -> dict[str, DataLoader]:
    """Return every split of a build in batches, whole tiles at a time.

    Args:
        build: The build the splits are cut from.
        sizes: How far a patch of each sensor runs along each axis it is cut on.
        pool: How many ground samples of a patch each instrument averages into one.
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
    by_tile = build.read_observation_metadata()
    axes = build.read_axes_by_instrument()
    return {
        name: tile_loader(
            build,
            {tile: by_tile[tile] for tile in held},
            axes,
            sizes,
            pool,
            shapes,
            delay,
            collate,
            batch_size,
            workers,
            name == TRAINING_SPLIT,
        )
        for name, held in split_tiles(by_tile, shares, seed).items()
    }


def tile_loader(
    build: DatasetBuild,
    tiles: Mapping[str, dict[str, list[ObservationMetadata]]],
    axes: Mapping[str, tuple[str, ...]],
    sizes: Mapping[str, Mapping[str, int]],
    pool: Mapping[str, int],
    shapes: Mapping[str, tuple[int, ...]],
    delay: str,
    collate: Callable,
    batch_size: int,
    workers: int,
    shuffle: bool,
) -> DataLoader:
    """Return some tiles of a build in batches, whole tiles at a time.

    Args:
        build: The build the tiles are read from.
        tiles: The index rows of each sensor of each tile, keyed by tile.
        axes: What each axis of each instrument's values holds.
        sizes: How far a patch of each sensor runs along each axis it is cut on.
        pool: How many ground samples of a patch each instrument averages into one.
        shapes: The shape of one patch of each instrument as the model reads it.
        delay: The instrument whose rows give every surface patch its delay.
        collate: How one batch of read tiles becomes what the model is handed.
        batch_size: How many tiles one batch holds.
        workers: How many processes read tiles beside the model.
        shuffle: Whether the tiles come in a new order every pass.

    Returns:
        loader: The tiles in batches.
    """
    gc.freeze()
    return DataLoader(
        TileDataset(build, tiles, axes, sizes, pool, shapes, delay),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=True,
        collate_fn=collate,
    )


class TileDataset(Dataset):
    """Some tiles of a build, each read as the patches of each instrument.

    Attributes:
        build: The build the tiles are read from.
        tiles: The index rows of each sensor of each tile, keyed by tile.
        identities: The tiles, in the order they are read.
        axes: What each axis of each instrument's values holds.
        sizes: How far a patch of each sensor runs along each axis it is cut on.
        pool: How many ground samples of a patch each instrument averages into one.
        shapes: The shape of one patch of each instrument as the model reads it.
        delay: The instrument whose rows give every surface patch its delay.
    """

    def __init__(
        self,
        build: DatasetBuild,
        tiles: Mapping[str, dict[str, list[ObservationMetadata]]],
        axes: Mapping[str, tuple[str, ...]],
        sizes: Mapping[str, Mapping[str, int]],
        pool: Mapping[str, int],
        shapes: Mapping[str, tuple[int, ...]],
        delay: str,
    ) -> None:
        """Keep what every read needs.

        Args:
            build: The build the tiles are read from.
            tiles: The index rows of each sensor of each tile, keyed by tile.
            axes: What each axis of each instrument's values holds.
            sizes: How far a patch of each sensor runs along each axis it is cut on.
            pool: How many ground samples of a patch each instrument averages into one.
            shapes: The shape of one patch of each instrument as the model reads it.
            delay: The instrument whose rows give every surface patch its delay.
        """
        self.build = build
        self.tiles = tiles
        self.identities = list(tiles)
        self.axes = axes
        self.sizes = sizes
        self.pool = pool
        self.shapes = shapes
        self.delay = delay

    def __len__(self) -> int:
        """Return how many tiles there are.

        Returns:
            count: One per tile.
        """
        return len(self.identities)

    def __getitem__(self, index: int) -> tuple[dict[str, dict[str, np.ndarray]], str]:
        """Return one tile's patches, read from the run's cache or cut and cached.

        Args:
            index: Which tile.

        Returns:
            sample: Each sensor's patches: values, valid and position.
            identity: The tile the read belongs to.
        """
        identity = self.identities[index]
        held = self.build.root / ready_tile_path(identity)
        if held.is_file():
            sample = {}
            with np.load(held) as arrays:
                for packed in arrays.files:
                    name, key = packed.split("/")
                    sample.setdefault(name, {})[key] = arrays[packed]
            return sample, identity
        rows = self.tiles[identity]
        delays = read_surface_delays(self.build, rows[self.delay])
        read = read_tile_patches(rows, self.build, self.sizes, self.pool, delays)
        sample = {
            name: patch_arrays(drawn, self.shapes[name], self.axes[name])
            for name, drawn in read.items()
        }
        with atomic_path(held) as written, written.open("wb") as file:
            np.savez_compressed(
                file,
                **{
                    f"{name}/{key}": array
                    for name, arrays in sample.items()
                    for key, array in arrays.items()
                },
            )
        return sample, identity


def patch_arrays(
    patches: Sequence[Patch],
    shape: Sequence[int],
    axes: Sequence[str],
) -> dict[str, np.ndarray]:
    """Return one instrument's read patches as the arrays a model is handed.

    Args:
        patches: The patches, all of one instrument.
        shape: The shape of one patch of it as the model reads it.
        axes: What each axis of its values holds.

    Returns:
        arrays: The patches under "values", "valid" and "position".
    """
    valid_shape = tuple(
        held if holds in (Axis.GROUND, Axis.WAVELENGTH) else 1
        for held, holds in zip(shape, axes, strict=True)
    )
    return {
        "values": stacked([one.values for one in patches], tuple(shape), np.float32),
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
    return np.stack(arrays).astype(dtype, copy=False)
