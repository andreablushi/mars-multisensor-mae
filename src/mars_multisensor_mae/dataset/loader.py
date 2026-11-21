"""The loaders reading a build's tiles in batches, and their cache."""

from __future__ import annotations

import gc
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial

import numpy as np
import torch
from mars_multisensor_dataset.building.common.layout import Axis
from mars_multisensor_dataset.building.configs import sharad
from mars_multisensor_dataset.common.disk.files import atomic_path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from mars_multisensor_mae.architecture.models.tokens import Tokens
from mars_multisensor_mae.configs.load import named_generator
from mars_multisensor_mae.configs.paths import ready_array_path
from mars_multisensor_mae.dataset.models.patch import (
    InstrumentPatches,
    PatchLayout,
    TileSample,
)
from mars_multisensor_mae.dataset.patches import cut_tile_patches
from mars_multisensor_mae.dataset.store import DatasetBuild


def padded_tokens(samples: Sequence[TileSample]) -> dict[str, Tokens]:
    """Return every instrument's patches over some tiles, padded to one count.

    Args:
        samples: Each tile's patches.

    Returns:
        batch: Each instrument's patches over the tiles, keyed as ODE names it.
    """
    batch = {}
    # Pad each instrument's patches across the batch's tiles
    for name in samples[0].instruments:
        # Every array a model reads padded with zeros to the most patches any tile has
        padded = {
            key: pad_sequence(
                [
                    torch.as_tensor(getattr(one.instruments[name], key))
                    for one in samples
                ],
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
) -> TileSample:
    """Return one tile's patches from the run's cache, cut into it when missing.

    Args:
        build: The build the tile is read from.
        identity: The tile.
        layout: How each instrument is cut into patches, and the shape the model
            reads one in.
        cache: Whether a tile missing from the cache is kept there once cut, or cut
            in memory alone.

    Returns:
        sample: Each instrument's patches as stored, read lazily off the cache.
    """
    paths = {
        name: {
            key: build.root
            / ready_array_path(
                identity,
                name,
                (
                    sorted(size.items()),
                    build.acquisition[name],
                    build.sharad_atmosphere_cropping
                    and name == sharad.LAYOUT.instrument,
                ),
                key,
            )
            for key in InstrumentPatches._fields
        }
        for name, size in layout.sizes.items()
    }
    rows = build.read_observation_metadata()[identity]
    if not all(path.is_file() for held in paths.values() for path in held.values()):
        cut = {
            name: InstrumentPatches.stacked(
                patches,
                name,
                layout.shapes[name],
                rows[name][0].axes,
                build.acquisition[name],
            )
            for name, patches in cut_tile_patches(rows, build, layout).items()
        }
        if not cache:
            return TileSample(cut)
        for name, held in cut.items():
            for key, array in held._asdict().items():
                if key == "measured":
                    array = np.packbits(array, axis=-1)
                with (
                    atomic_path(paths[name][key]) as written,
                    written.open("wb") as file,
                ):
                    np.save(file, array)
    loaded = {}
    for name, held in paths.items():
        arrays = {key: np.load(path, mmap_mode="c") for key, path in held.items()}
        # What is measured is kept a bit a sample, along the last axis it runs on
        last = layout.shapes[name][-1] if rows[name][0].axes[-1] == Axis.GROUND else 1
        arrays["measured"] = np.unpackbits(
            arrays["measured"], axis=-1, count=last
        ).view(bool)
        loaded[name] = InstrumentPatches(**arrays)
    return TileSample(loaded)


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
            grounds = sample.ground_spots
            centre = grounds[draw.randrange(len(grounds))]
            sample = sample.budget_cut(budget, centre)
        samples.append(build.normalised_sample(sample))
    return padded_tokens(samples), list(identities)


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
    read_batch: Callable[..., tuple] | None = None,
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
        read_batch: What reads one batch from its tiles, the build, the layout, the
            budget and the cache flag, in place of one budget cut a tile, or None.

    Returns:
        loader: The tiles in batches, as the batch reader hands them.
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
            read_batch, build=build, layout=layout, budget=budget, cache=cache
        )
        if read_batch
        else partial(
            read_cached_batch,
            build=build,
            layout=layout,
            budget=budget,
            scale=budget_scale if training else 1.0,
            seed=None if training else seed,
            cache=cache,
        ),
    )
