"""The loaders reading a build's tiles in batches, and their cache."""

from __future__ import annotations

import gc
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from mars_multisensor_dataset.building.common.layout import Axis
from mars_multisensor_dataset.building.configs import sharad
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from mars_multisensor_mae.architecture.models.tokens import Tokens
from mars_multisensor_mae.configs.load import named_generator
from mars_multisensor_mae.configs.paths import (
    READY_MARKER,
    READY_ROOT_NAME,
    READY_TILE_FILE,
    ready_setup_folder,
)
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


def ready_setup_folders(build: DatasetBuild, layout: PatchLayout) -> dict[str, Path]:
    """Return the folder each instrument's tiles of a build are cut into.

    Args:
        build: The build the tiles are read from.
        layout: How each instrument is cut into patches, which names its folder.

    Returns:
        folders: The folder of each instrument's cut tiles, keyed as ODE names it.
    """
    held = build.ready_root or build.root / READY_ROOT_NAME
    return {
        name: held
        / ready_setup_folder(
            name,
            (
                sorted(size.items()),
                build.acquisition[name],
                build.sharad_atmosphere_cropping and name == sharad.LAYOUT.instrument,
            ),
        )
        for name, size in layout.sizes.items()
    }


def ready_tile_paths(
    build: DatasetBuild, identity: str, layout: PatchLayout
) -> dict[str, Path]:
    """Return where each instrument's cut patches of one tile are kept.

    Args:
        build: The build the tile is read from.
        identity: The tile.
        layout: How each instrument is cut into patches, which names its folder.

    Returns:
        paths: One uncompressed npz per instrument, in the band its crops sit in.
    """
    rows = build.read_observation_metadata()[identity]
    return {
        name: folder
        / READY_TILE_FILE.format(PurePosixPath(rows[name][0].path).parts[0], identity)
        for name, folder in ready_setup_folders(build, layout).items()
    }


def cut_tile_sample(
    build: DatasetBuild, identity: str, layout: PatchLayout
) -> TileSample:
    """Return one tile's patches, cut from its observations as stored.

    Args:
        build: The build the tile is read from.
        identity: The tile.
        layout: How each instrument is cut into patches, and the shape the model
            reads one in.

    Returns:
        sample: Each instrument's patches as stored, in memory.
    """
    rows = build.read_observation_metadata()[identity]
    return TileSample(
        {
            name: InstrumentPatches.stacked(
                patches,
                name,
                layout.shapes[name],
                rows[name][0].axes,
                build.acquisition[name],
            )
            for name, patches in cut_tile_patches(rows, build, layout).items()
        }
    )


def write_cut_tile(sample: TileSample, paths: Mapping[str, Path]) -> None:
    """Write each instrument's patches of one tile to its own file, renamed whole.

    Args:
        sample: Each instrument's patches as stored.
        paths: Where each instrument's patches are written, keyed as ODE names it.
    """
    for name, held in sample.instruments.items():
        arrays = held._asdict()
        arrays["measured"] = np.packbits(arrays["measured"], axis=-1)
        path = paths[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        unfinished = path.with_name(f"{path.name}.partial")
        with unfinished.open("wb") as file:
            np.savez(file, **arrays)
        unfinished.replace(path)


def cut_missing_tiles(
    identities: Sequence[str], build: DatasetBuild, layout: PatchLayout
) -> int:
    """Return how many of some tiles were cut, each one missing cut and written.

    Args:
        identities: The tiles.
        build: The build the tiles are read from.
        layout: How each instrument is cut into patches, and the shape the model
            reads one in.

    Returns:
        cut: How many of the tiles were missing and are now cut.
    """
    cut = 0
    for identity in identities:
        paths = ready_tile_paths(build, identity, layout)
        if not all(path.is_file() for path in paths.values()):
            write_cut_tile(cut_tile_sample(build, identity, layout), paths)
            cut += 1
    return cut


def check_complete_cut(build: DatasetBuild, layout: PatchLayout) -> None:
    """Check that a build which never cuts holds a complete cut of each instrument.

    Args:
        build: The build the tiles are read from.
        layout: How each instrument is cut into patches, which names its folder.

    Raises:
        FileNotFoundError: When the cut of an instrument was never completed.
    """
    if build.cuts:
        return
    missing = [
        str(folder)
        for folder in ready_setup_folders(build, layout).values()
        if not (folder / READY_MARKER).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"no complete cut in {', '.join(missing)}: cut it with "
            "`scripts/leonardo/submit.sh cut` and the stage's own overrides"
        )


def read_cached_tile(
    build: DatasetBuild,
    identity: str,
    layout: PatchLayout,
    cache: bool = True,
) -> TileSample:
    """Return one tile's patches from the cut tiles, cut into them when missing.

    Args:
        build: The build the tile is read from.
        identity: The tile.
        layout: How each instrument is cut into patches, and the shape the model
            reads one in.
        cache: Whether a tile missing from the cut tiles is kept there once cut, or
            cut in memory alone.

    Returns:
        sample: Each instrument's patches as stored, read off the cut tiles.

    Raises:
        FileNotFoundError: When the tile is missing and the build never cuts.
    """
    paths = ready_tile_paths(build, identity, layout)
    if not all(path.is_file() for path in paths.values()):
        if not build.cuts:
            raise FileNotFoundError(f"{identity} is not cut: {sorted(paths.values())}")
        sample = cut_tile_sample(build, identity, layout)
        if not cache:
            return sample
        write_cut_tile(sample, paths)
    rows = build.read_observation_metadata()[identity]
    loaded = {}
    for name, path in paths.items():
        with np.load(path) as held:
            arrays = {key: held[key] for key in InstrumentPatches._fields}
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
