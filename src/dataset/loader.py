"""The splits of a build, and the loaders reading their tiles in batches."""

from __future__ import annotations

import gc
import random
from collections.abc import Mapping, Sequence
from functools import partial

import torch
from torch.utils.data import DataLoader

from configs.load import named_seed
from dataset.store import DatasetBuild

TRAINING_SPLIT = "train"
VALIDATION_SPLIT = "validation"
SPLITS = (TRAINING_SPLIT, VALIDATION_SPLIT)


def loaders_by_split(
    build: DatasetBuild,
    sizes: Mapping[str, Mapping[str, int]],
    budget: Mapping[str, int],
    budget_scale: float,
    shares: Sequence[float],
    seed: int,
    batch_size: int,
    workers: int,
) -> dict[str, DataLoader]:
    """Return every split of a build in batches, each tile cut to the budget.

    Args:
        build: The build the splits are cut from.
        sizes: How far a patch of each sensor runs along each axis it is cut on.
        budget: How many patches of each instrument a tile keeps.
        budget_scale: The most a training batch's budget is multiplied by.
        shares: The share of the observations each split holds, in the code's order.
        seed: What fixes which split a tile falls in, each validation tile's cut,
            and the training's order and cuts.
        batch_size: How many tiles one batch holds.
        workers: How many processes read tiles beside the training.

    Returns:
        loaders: One loader per split, the training one shuffled and the other not.
    """
    by_tile = build.read_observation_metadata()
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
        name = min(SPLITS, key=lambda one: placed[one] / wanted[one])
        splits[name].append(tile)
        placed[name] += counted[tile]
    return {
        name: tile_loader(
            build,
            held,
            sizes,
            batch_size,
            workers,
            name == TRAINING_SPLIT,
            budget,
            budget_scale if name == TRAINING_SPLIT else 1.0,
            None if name == TRAINING_SPLIT else seed,
            torch.Generator().manual_seed(named_seed(seed, f"loader/{name}")),
        )
        for name, held in splits.items()
    }


def tile_loader(
    build: DatasetBuild,
    identities: Sequence[str],
    sizes: Mapping[str, Mapping[str, int]],
    batch_size: int,
    workers: int,
    shuffle: bool,
    budget: Mapping[str, int] | None,
    scale: float,
    seed: int | None,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Return some cached tiles of a build in batches.

    Args:
        build: The build the tiles are read from.
        identities: The tiles.
        sizes: How far a patch of each sensor runs along each axis it is cut on.
        batch_size: How many tiles one batch holds.
        workers: How many processes read tiles beside the model.
        shuffle: Whether the tiles come in a new order every pass.
        budget: How many patches of each instrument a tile keeps, or None to read
            it whole.
        scale: The most the budget is multiplied by, drawn once per batch from 1.
        seed: What fixes each tile's cut, or None to draw a new one each read.
        generator: What fixes the order and the readers' own draws, or None to
            take them from the global stream.

    Returns:
        loader: The tiles in batches.
    """
    gc.freeze()
    return DataLoader(
        list(identities),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=True,
        generator=generator,
        collate_fn=partial(
            build.read_batch, sizes=sizes, budget=budget, scale=scale, seed=seed
        ),
    )


def cache_tiles(
    build: DatasetBuild,
    sizes: Mapping[str, Mapping[str, int]],
    shapes: Mapping[str, tuple[int, ...]],
    workers: int,
) -> None:
    """Cut every tile of a build into the run's cache, before any of it is read.

    Args:
        build: The build whose tiles are cut.
        sizes: How far a patch of each sensor runs along each axis it is cut on.
        shapes: The shape of one patch of each instrument as the model reads it.
        workers: How many processes cut tiles.
    """
    by_tile = build.read_observation_metadata()
    # Its own generator leaves the training's shuffle as it would be without the fill
    for _ in DataLoader(
        list(by_tile),
        batch_size=None,
        num_workers=workers,
        collate_fn=lambda identity: build.cache_tile(
            identity, by_tile[identity], sizes, shapes
        ),
        generator=torch.Generator(),
    ):
        pass
