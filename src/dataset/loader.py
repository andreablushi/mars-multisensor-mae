"""The loaders reading a build's tiles in batches, and the cache they read."""

from __future__ import annotations

import gc
from collections.abc import Iterable, Mapping
from functools import partial

import torch
from torch.utils.data import DataLoader

from configs.load import named_seed
from dataset.store import DatasetBuild


def tile_loader(
    build: DatasetBuild,
    tiles: Iterable[str],
    sizes: Mapping[str, Mapping[str, int]],
    budget: Mapping[str, int] | None,
    budget_scale: float,
    seed: int,
    batch_size: int,
    workers: int,
    training: bool,
) -> DataLoader:
    """Return some cached tiles of a build in batches.

    Args:
        build: The build the tiles are read from.
        tiles: The tiles.
        sizes: How far a patch of each sensor runs along each axis it is cut on.
        budget: How many patches of each instrument a tile keeps, or None for all.
        budget_scale: The most a training batch's budget is multiplied by.
        seed: What fixes the order, the readers' own draws and each fixed cut.
        batch_size: How many tiles one batch holds.
        workers: How many processes read tiles beside the model.
        training: Whether the tiles are shuffled, scaled and cut anew every pass.

    Returns:
        loader: The tiles in batches.
    """
    gc.freeze()
    return DataLoader(
        list(tiles),
        batch_size=batch_size,
        shuffle=training,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=True,
        generator=torch.Generator().manual_seed(
            named_seed(seed, "loader/train" if training else "loader/validation")
        ),
        collate_fn=partial(
            build.read_batch,
            sizes=sizes,
            budget=budget,
            scale=budget_scale if training else 1.0,
            seed=None if training else seed,
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
