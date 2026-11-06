"""The loaders reading a build's tiles in batches, the cache they read, and splits."""

from __future__ import annotations

import gc
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from functools import partial

import numpy as np
import torch
from analysis.ground_truth.swim.models import IceLabel
from building.configs import sharad
from common.disk.files import atomic_path
from sklearn.model_selection import StratifiedGroupKFold
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from architecture.models.tokens import Tokens
from configs.load import named_generator, named_seed
from configs.paths import ready_array_path
from dataset.patches import cut_tile_patches, patch_arrays
from dataset.store import DatasetBuild


def padded_tokens(
    samples: Sequence[Mapping[str, Mapping[str, np.ndarray]]],
) -> dict[str, Tokens]:
    """Return every instrument's patches over some tiles, padded to one count.

    Args:
        samples: Each tile's patch arrays of each instrument.

    Returns:
        batch: Each instrument's patches over the tiles, keyed as ODE names it.
    """
    batch = {}
    # Pad each instrument's patches across the batch's tiles
    for name in samples[0]:
        # Every array a model reads padded with zeros to the most patches any tile has
        batch[name] = Tokens(
            **{
                key: pad_sequence(
                    [torch.as_tensor(one[name][key]) for one in samples],
                    batch_first=True,
                )
                for key in Tokens._fields
            }
        )
    return batch


def read_cached_tile(
    build: DatasetBuild,
    identity: str,
    sizes: Mapping[str, Mapping[str, int]],
    shapes: Mapping[str, tuple[int, ...]],
) -> dict[str, dict[str, np.ndarray]]:
    """Return one tile's patch arrays from the run's cache, cut into it when missing.

    Args:
        build: The build the tile is read from.
        identity: The tile.
        sizes: How far a patch of each sensor runs along each axis it is cut on.
        shapes: The shape of one patch of each instrument as the model reads it.

    Returns:
        sample: Each instrument's patch arrays, keyed as ODE names it.
    """
    paths = {
        name: {
            key: build.root
            / ready_array_path(
                identity,
                name,
                (
                    sorted(size.items()),
                    build.clips[name],
                    build.acquisition[name],
                    build.sharad_atmosphere_cropping
                    and name == sharad.LAYOUT.instrument,
                ),
                key,
            )
            for key in Tokens._fields
        }
        for name, size in sizes.items()
    }
    if not all(path.is_file() for held in paths.values() for path in held.values()):
        rows = build.read_observation_metadata()[identity]
        for name, patches in cut_tile_patches(rows, build, sizes).items():
            arrays = patch_arrays(
                patches, name, shapes[name], rows[name][0].axes, build.acquisition[name]
            )
            for key, array in arrays.items():
                with (
                    atomic_path(paths[name][key]) as written,
                    written.open("wb") as file,
                ):
                    np.save(file, array)
    return {
        name: {key: np.load(path, mmap_mode="c") for key, path in held.items()}
        for name, held in paths.items()
    }


def read_cached_batch(
    identities: Sequence[str],
    build: DatasetBuild,
    sizes: Mapping[str, Mapping[str, int]],
    shapes: Mapping[str, tuple[int, ...]],
    budget: Mapping[str, int] | None,
    scale: float,
    seed: int | None,
) -> tuple[dict[str, Tokens], list[str]]:
    """Return one batch of cached tiles, each cut to the budget around a centre.

    Args:
        identities: The tiles of the batch.
        build: The build the tiles are read from.
        sizes: How far a patch of each sensor runs along each axis it is cut on.
        shapes: The shape of one patch of each instrument as the model reads it.
        budget: How many patches of each instrument a tile keeps, or None to
            read it whole.
        scale: The most the budget is multiplied by, drawn once per batch from 1.
        seed: What fixes each tile's centre, or None to draw a new one each read.

    Returns:
        batch: Each instrument's patches over the batch, keyed as ODE names it.
        identities: The tile each read belongs to, in the batch's own order.
    """
    if budget is not None:
        factor = random.uniform(1.0, scale)
        budget = {name: round(count * factor) for name, count in budget.items()}
    samples = []
    for identity in identities:
        sample = read_cached_tile(build, identity, sizes, shapes)
        if budget is not None:
            draw = random.Random(
                random.random() if seed is None else f"{seed}/{identity}"
            )
            # Each ground spot once, whether a sounder's delay is cut or not
            grounds = np.unique(
                np.concatenate([held["position"][:, :2] for held in sample.values()]),
                axis=0,
            )
            # One patch spot of any instrument is the centre all are cut around
            centre = grounds[draw.randrange(len(grounds))]
            for name, held in sample.items():
                apart = np.linalg.norm(held["position"][:, :2] - centre, axis=1)
                kept = np.argsort(apart, kind="stable")[: budget[name]]
                sample[name] = {key: array[kept] for key, array in held.items()}
        samples.append(sample)
    return padded_tokens(samples), list(identities)


def tile_loader(
    build: DatasetBuild,
    tiles: Iterable[str],
    sizes: Mapping[str, Mapping[str, int]],
    shapes: Mapping[str, tuple[int, ...]],
    budget: Mapping[str, int] | None,
    budget_scale: float,
    seed: int,
    batch_size: int,
    workers: int,
    training: bool,
) -> DataLoader:
    """Return some tiles of a build in batches, each cut into the cache on first read.

    Args:
        build: The build the tiles are read from.
        tiles: The tiles.
        sizes: How far a patch of each sensor runs along each axis it is cut on.
        shapes: The shape of one patch of each instrument as the model reads it.
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
        generator=named_generator(
            seed, "loader/train" if training else "loader/validation"
        ),
        collate_fn=partial(
            read_cached_batch,
            build=build,
            sizes=sizes,
            shapes=shapes,
            budget=budget,
            scale=budget_scale if training else 1.0,
            seed=None if training else seed,
        ),
    )


def ice_split(
    labels: Mapping[str, IceLabel], folds: int, block_deg: float, seed: int
) -> tuple[list[str], list[str], list[str]]:
    """Return the labelled tiles split by place into training, validation and test.

    Args:
        labels: Each tile's ice at both depths, keyed by the tile.
        folds: How many equal folds the tiles are dealt into, the first testing, the
            second validating and the rest training.
        block_deg: How wide a square of the map is, in degrees, each one dealt whole.
        seed: The run's seed, which fixes the deal.

    Returns:
        training: The training tiles.
        validation: The validation tiles.
        test: The test tiles.
    """
    tiles = sorted(labels)
    squares = [
        (
            math.floor(labels[tile].centre_lat / block_deg),
            math.floor(labels[tile].centre_lon / block_deg),
        )
        for tile in tiles
    ]
    strata = [(labels[tile].ice_1_5m, labels[tile].ice_5m) for tile in tiles]
    dealt = StratifiedGroupKFold(
        folds, shuffle=True, random_state=named_seed(seed, "benchmark/split")
    ).split(tiles, [str(one) for one in strata], [str(one) for one in squares])
    test, validation, *rest = (held for _, held in dealt)
    return tuple(
        [tiles[at] for at in sorted(held)]
        for held in (np.concatenate(rest), validation, test)
    )
