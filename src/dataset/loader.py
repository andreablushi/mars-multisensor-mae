"""The loaders reading a build's tiles in batches, their cache, splits and targets."""

from __future__ import annotations

import gc
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from functools import partial

import numpy as np
import torch
from ice_consistency.analysis.position import grid_offsets_from_tile_centre
from ice_consistency.models import IceLabel
from mars_multisensor_dataset.building.common.layout import Axis
from mars_multisensor_dataset.building.configs import mola, sharad
from mars_multisensor_dataset.common.disk.files import atomic_path
from sklearn.model_selection import StratifiedGroupKFold
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from architecture.models.tokens import Tokens
from configs.load import named_generator, named_seed
from configs.paths import ready_array_path
from dataset.models.patch import PatchLayout
from dataset.patches import (
    cut_tile_patches,
    patch_arrays,
    patch_budget_selection,
    patch_ground_spots,
)
from dataset.store import DatasetBuild

HALVED = ("values", "condition")

TARGETS = {
    "1_5m": "consistency_1_5m",
    "5m": "consistency_5m",
    "rs": "radar_surface_power",
}


def padded_tokens(
    samples: Sequence[Mapping[str, Mapping[str, np.ndarray]]],
) -> dict[str, Tokens]:
    """Return every instrument's patches over some tiles, padded to one count.

    Args:
        samples: Each tile's patch arrays of each instrument.

    Returns:
        batch: Each instrument's patches over the tiles, keyed as ODE names it, what
            the cache holds in half precision kept so until the device.
    """
    batch = {}
    # Pad each instrument's patches across the batch's tiles
    for name in samples[0]:
        # Every array a model reads padded with zeros to the most patches any tile has
        padded = {
            key: pad_sequence(
                [torch.as_tensor(one[name][key]) for one in samples],
                batch_first=True,
            )
            for key in Tokens._fields
        }
        batch[name] = Tokens(**padded)
    return batch


def read_cached_tile(
    build: DatasetBuild,
    identity: str,
    layout: PatchLayout,
    cache: bool = True,
) -> dict[str, dict[str, np.ndarray]]:
    """Return one tile's patch arrays from the run's cache, cut into it when missing.

    Args:
        build: The build the tile is read from.
        identity: The tile.
        layout: How each instrument is cut into patches, and the shape the model
            reads one in.
        cache: Whether a tile missing from the cache is kept there once cut, or cut
            in memory alone.

    Returns:
        sample: Each instrument's patch arrays, keyed as ODE names it, values and
            condition in the build's half precision except MOLA's elevations.
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
        for name, size in layout.sizes.items()
    }
    rows = build.read_observation_metadata()[identity]
    if not all(path.is_file() for held in paths.values() for path in held.values()):
        cut = {
            name: {
                key: array.astype(np.float16)
                if key in HALVED and name != mola.LAYOUT.instrument
                else array
                for key, array in patch_arrays(
                    patches,
                    name,
                    layout.shapes[name],
                    rows[name][0].axes,
                    build.acquisition[name],
                ).items()
            }
            for name, patches in cut_tile_patches(rows, build, layout.sizes).items()
        }
        if not cache:
            return cut
        for name, arrays in cut.items():
            for key, array in arrays.items():
                if key == "measured":
                    array = np.packbits(array, axis=-1)
                with (
                    atomic_path(paths[name][key]) as written,
                    written.open("wb") as file,
                ):
                    np.save(file, array)
    sample = {
        name: {key: np.load(path, mmap_mode="c") for key, path in held.items()}
        for name, held in paths.items()
    }
    for name, held in sample.items():
        # What is measured is kept a bit a sample, along the last axis it runs on
        last = layout.shapes[name][-1] if rows[name][0].axes[-1] == Axis.GROUND else 1
        held["measured"] = np.unpackbits(held["measured"], axis=-1, count=last).view(
            bool
        )
    return sample


def read_cached_batch(
    identities: Sequence[str],
    build: DatasetBuild,
    layout: PatchLayout,
    budget: Mapping[str, int] | None,
    scale: float,
    seed: int | None,
    cache: bool = True,
) -> tuple[dict[str, Tokens], list[str]]:
    """Return one batch of cached tiles, each cut to the budget around a centre.

    Args:
        identities: The tiles of the batch.
        build: The build the tiles are read from.
        layout: How each instrument is cut into patches, and the shape the model
            reads one in.
        budget: How many patches of each instrument a tile keeps, or None to
            read it whole.
        scale: The most the budget is multiplied by, drawn once per batch from 1.
        seed: What fixes each tile's centre, or None to draw a new one each read.
        cache: Whether a tile missing from the cache is kept there once cut.

    Returns:
        batch: Each instrument's patches over the batch, keyed as ODE names it.
        identities: The tile each read belongs to, in the batch's own order.
    """
    if budget is not None:
        factor = random.uniform(1.0, scale)
        budget = {name: round(count * factor) for name, count in budget.items()}
    samples = []
    for identity in identities:
        sample = read_cached_tile(build, identity, layout, cache)
        if budget is not None:
            draw = random.Random(
                random.random() if seed is None else f"{seed}/{identity}"
            )
            # One patch spot of any instrument is the centre all are cut around
            grounds = patch_ground_spots(sample)
            centre = grounds[draw.randrange(len(grounds))]
            sample = patch_budget_selection(sample, budget, centre)
        samples.append(sample)
    return padded_tokens(samples), list(identities)


def measured_patch_spots(sample: Mapping[str, Mapping[str, np.ndarray]]) -> np.ndarray:
    """Return the ground spot of every patch holding a measurement, any instrument.

    Args:
        sample: Each instrument's patch arrays of one tile or crop.

    Returns:
        spots: East and north of each such patch in metres. (S, 2)
    """
    return np.concatenate(
        [
            held["position"][
                held["measured"].any(tuple(range(1, held["measured"].ndim))), :2
            ]
            for held in sample.values()
        ]
    )


def read_covering_crops_batch(
    identities: Sequence[str],
    build: DatasetBuild,
    layout: PatchLayout,
    budget: Mapping[str, int],
    reach_m: float,
    cache: bool = True,
) -> tuple[dict[str, Tokens], list[str], np.ndarray]:
    """Return each tile of a batch as the budget crops that cover SWIM's points.

    Args:
        identities: The tiles of the batch.
        build: The build the tiles are read from.
        layout: How each instrument is cut into patches, and the shape the model
            reads one in.
        budget: How many patches of each instrument a crop keeps.
        reach_m: How near a read patch a point is scored, in metres.
        cache: Whether a tile missing from the cache is kept there once cut.

    Returns:
        batch: Each instrument's patches over every crop, keyed as ODE names it.
        identities: The tile each crop belongs to, a tile's crops side by side.
        scored: Which points each crop predicts: those within reach of its read
            patches, and those of no crop in the crop centred nearest them. (C, P * P)
    """
    points = np.stack(grid_offsets_from_tile_centre(), -1).reshape(-1, 2)  # (P * P, 2)
    samples, tiles, scored = [], [], []
    for identity in identities:
        sample = read_cached_tile(build, identity, layout, cache)
        grounds = measured_patch_spots(sample)  # (G, 2)
        apart = np.linalg.norm(points[:, None] - grounds[None], axis=-1)  # (P * P, G)
        left = apart.min(1) <= reach_m
        centres, reached = [], []
        # Each crop is centred on the spot nearest the first point no crop reaches yet
        while left.any() or not centres:
            first = int(np.argmax(left))
            centre = grounds[apart[first].argmin()]
            crop = patch_budget_selection(sample, budget, centre)
            spots = measured_patch_spots(crop)
            near = (
                np.linalg.norm(points[:, None] - spots[None], axis=-1).min(1) <= reach_m
            )
            near[first] = True
            left &= ~near
            centres.append(centre)
            reached.append(near)
            samples.append(crop)
        reached = np.stack(reached)  # (C, P * P)
        # A point no crop reaches is predicted by the crop centred nearest it
        lone = ~reached.any(0)
        nearest = np.linalg.norm(points[:, None] - np.stack(centres)[None], axis=-1)
        reached[nearest.argmin(1)[lone], np.flatnonzero(lone)] = True
        tiles += [identity] * len(reached)
        scored.append(reached)
    return padded_tokens(samples), tiles, np.concatenate(scored)


def tile_loader(
    build: DatasetBuild,
    tiles: Iterable[str],
    layout: PatchLayout,
    budget: Mapping[str, int] | None,
    budget_scale: float,
    seed: int,
    batch_size: int,
    workers: int,
    training: bool,
    drop_last: bool = False,
    cache: bool = True,
    reach_m: float | None = None,
) -> DataLoader:
    """Return some tiles of a build in batches, each cut into the cache on first read.

    Args:
        build: The build the tiles are read from.
        tiles: The tiles.
        layout: How each instrument is cut into patches, and the shape the model
            reads one in.
        budget: How many patches of each instrument a tile keeps, or None for all.
        budget_scale: The most a training batch's budget is multiplied by.
        seed: What fixes the order, the readers' own draws and each fixed cut.
        batch_size: How many tiles one batch holds.
        workers: How many processes read tiles beside the model.
        training: Whether the tiles are shuffled, scaled and cut anew every pass.
        drop_last: Whether a last batch short of batch_size is left out.
        cache: Whether a tile missing from the cache is kept there once cut.
        reach_m: How near a read patch a point is scored, to read each tile as the
            budget crops covering SWIM's points, or None for one read a tile.

    Returns:
        loader: The tiles in batches, each crop with the points it predicts when
            read as covering crops.
    """
    gc.freeze()
    return DataLoader(
        list(tiles),
        batch_size=batch_size,
        shuffle=training,
        drop_last=drop_last,
        num_workers=workers,
        persistent_workers=workers > 0,
        prefetch_factor=1 if workers else None,
        pin_memory=True,
        generator=named_generator(
            seed, "loader/train" if training else "loader/validation"
        ),
        collate_fn=partial(
            read_cached_batch,
            build=build,
            layout=layout,
            budget=budget,
            scale=budget_scale if training else 1.0,
            seed=None if training else seed,
            cache=cache,
        )
        if reach_m is None
        else partial(
            read_covering_crops_batch,
            build=build,
            layout=layout,
            budget=budget,
            reach_m=reach_m,
            cache=cache,
        ),
    )


def tiles_split_by_square(
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


def swim_targets_loader(labels: Mapping[str, IceLabel]) -> dict[str, np.ndarray]:
    """Return SWIM's value of every point of each tile for each target.

    Args:
        labels: Each tile's ice at both depths and its radar, keyed by the tile.

    Returns:
        targets: The values in TARGETS order, rows from the north and points from
            the west, keyed by the tile. (P * P, Z)
    """
    return {
        tile: np.stack([getattr(one, field) for field in TARGETS.values()], -1).astype(
            np.float32
        )
        for tile, one in labels.items()
    }
