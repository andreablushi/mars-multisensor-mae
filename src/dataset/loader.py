"""The splits of a build, and the loaders reading their tiles in batches."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from building.metadata.observation import ObservationMetadata
from torch.utils.data import DataLoader

from dataset.models.split import DatasetSplit
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
            key=lambda one: placed[one] / wanted[one] if wanted[one] else math.inf,
        )
        splits[name].append(tile)
        placed[name] += counted[tile]
    return splits


def read_training_statistics(
    build: DatasetBuild, shares: Sequence[float], seed: int
) -> dict[str, dict[str, np.ndarray]]:
    """Return what each instrument's values run to over the training split alone.

    Args:
        build: The build the split is cut from.
        shares: The share of the observations each split holds, in the code's order.
        seed: What fixes which split a tile falls in.

    Returns:
        statistics: Per sensor, the mean and deviation of its measurements.
    """
    splits = split_tiles(build.read_observation_metadata_by_tile(), shares, seed)
    return build.read_statistics_by_instrument(set(splits[TRAINING_SPLIT]))


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
    by_tile = build.read_observation_metadata_by_tile()
    splits = split_tiles(by_tile, shares, seed)
    statistics = build.read_statistics_by_instrument(set(splits[TRAINING_SPLIT]))
    axes = build.read_axes_by_instrument()
    DatasetSplit(
        build, by_tile, axes, statistics, sizes, pool, shapes, delay
    ).keep_every_tile(workers)
    return {
        name: DataLoader(
            DatasetSplit(
                build,
                {tile: by_tile[tile] for tile in held},
                axes,
                statistics,
                sizes,
                pool,
                shapes,
                delay,
            ),
            batch_size=batch_size,
            shuffle=name == TRAINING_SPLIT,
            num_workers=workers,
            persistent_workers=workers > 0,
            pin_memory=True,
            collate_fn=collate,
        )
        for name, held in splits.items()
    }
