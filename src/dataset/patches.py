"""Cutting one observation into the patches a transformer reads it as."""

from __future__ import annotations

import dataclasses
import math
import random
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from building.common.layout import Axis
from building.metadata.observation import ObservationMetadata

from dataset.bands import (
    measures_read_bands,
    observation_read_bands,
    read_band_patchsize,
)
from dataset.models.observation import Observation
from dataset.models.patch import Patch
from dataset.models.positioning import metres_from_centre

if TYPE_CHECKING:
    from dataset.store import DatasetBuild


def read_tile_patches(
    rows: Mapping[str, Sequence[ObservationMetadata]],
    build: DatasetBuild,
    sizes: Mapping[str, Mapping[str, int]],
    pool: Mapping[str, int],
    delays: np.ndarray,
) -> dict[str, list[Patch]]:
    """Return every patch of every observation one tile holds, by instrument.

    Args:
        rows: The tile's index rows of each instrument, keyed as ODE names it.
        build: The published build the observations are read from.
        sizes: How far a patch of each instrument runs along each axis it is cut on.
        pool: How many ground samples of a patch each instrument averages into one.
        delays: Which delay row the ground sounds at, over the tile. (N, 3)

    Returns:
        read: The patches of each instrument, none empty, keyed as ODE names it.
    """
    read: dict[str, list[Patch]] = {}
    for name, size in sizes.items():
        held: list[Patch] = []
        for record in rows.get(name, ()):
            # An observation missing any band read is left out whole
            if not measures_read_bands(name, record):
                continue
            observation = observation_read_bands(
                name, build.read_observation(record.path)
            )
            shape = observation.values.shape
            lengths = patch_lengths(shape, record.axes, size)
            counts = tuple(
                length // patch for length, patch in zip(shape, lengths, strict=True)
            )
            for at in range(math.prod(counts)):
                patch = cut_patch(observation, at, lengths, counts, delays)
                if patch.measured.any():
                    if name in pool:
                        patch = downsampled_patch(patch, pool[name])
                    held.append(patch)
        read[name] = held
    return read


def cut_patch(
    observation: Observation,
    index: int,
    lengths: tuple[int, ...],
    counts: tuple[int, ...],
    delays: np.ndarray,
) -> Patch:
    """Return one whole patch of one observation.

    Args:
        observation: The observation, read whole.
        index: Which patch, counting whole ones as the axes run, the last fastest.
        lengths: How far one patch of it runs along each axis.
        counts: How many whole patches each of those axes holds.
        delays: Which delay row the ground sounds at, over the tile. (N, 3)

    Returns:
        patch: The patch, copied, with what it measured and where it sits.
    """
    axes = observation.axes
    origin = tuple(
        int(at) * length
        for at, length in zip(np.unravel_index(index, counts), lengths, strict=True)
    )
    window = tuple(
        slice(start, start + length)
        for start, length in zip(origin, lengths, strict=True)
    )
    taken = tuple(window[at] for at in observation.ground_axes)
    # A separable grid is regular, so its corners place the patch as all its samples do
    corners = tuple(
        slice(one.start, one.stop, one.stop - one.start - 1 or 1) for one in taken
    )
    north, east = metres_from_centre(
        observation, corners if observation.described["separable"] else taken
    )
    north_m, east_m = float(np.mean(north)), float(np.mean(east))
    ground_shape = tuple(
        length if holds == Axis.GROUND else 1
        for length, holds in zip(lengths, axes, strict=True)
    )
    measured = observation.measured[taken].reshape(ground_shape).copy()
    if Axis.DELAY in axes:
        # A sounder is placed by the rows it sounded, which is the patch's own cut.
        at = axes.index(Axis.DELAY)
        delay = origin[at] + lengths[at] / 2
    else:
        # A patch on the ground sounds at one row, the one its surface echo lands on.
        nearest = np.argmin(
            (delays[:, 0] - north_m) ** 2 + (delays[:, 1] - east_m) ** 2
        )
        delay = float(delays[nearest, 2]) + 0.5
    return Patch(
        values=observation.values[window].copy(),
        measured=measured,
        axes=axes,
        north_m=north_m,
        east_m=east_m,
        delay=delay,
    )


def downsampled_patch(patch: Patch, factor: int) -> Patch:
    """Return a patch with every run of ground samples averaged into one.

    Args:
        patch: The patch, as cut from its observation.
        factor: How many ground samples are averaged into one, along each ground axis.

    Returns:
        patch: The patch, its unmeasured samples left out of every average.
    """
    ground = [at for at, holds in enumerate(patch.axes) if holds == Axis.GROUND]

    def split(shape: Sequence[int]) -> tuple[int, ...]:
        return tuple(
            length
            for at, size in enumerate(shape)
            for length in ((size // factor, factor) if at in ground else (size,))
        )

    # Where each ground axis's averaged samples land once it is split in two
    pooled = tuple(at + order + 1 for order, at in enumerate(ground))
    measured = patch.measured.reshape(split(patch.measured.shape))
    weight = np.broadcast_to(measured, split(patch.values.shape)).astype(np.float32)
    values = patch.values.reshape(weight.shape)
    return dataclasses.replace(
        patch,
        values=(values * weight).sum(axis=pooled)
        / np.maximum(weight.sum(axis=pooled), 1.0),
        measured=measured.any(axis=pooled),
    )


def patch_lengths(
    shape: Sequence[int],
    axes: Sequence[str],
    patchsize: Mapping[str, int],
    factor: int = 1,
) -> tuple[int, ...]:
    """Return how far one patch runs along each axis of an observation.

    Args:
        shape: How many samples each axis of the values holds.
        axes: What each of those axes holds, in that same order.
        patchsize: How far a patch runs along an axis holding each thing.
        factor: How many ground samples are averaged into one, 1 for the cut itself.

    Returns:
        lengths: One length per axis, an axis the patchsize omits taken whole.
    """
    return tuple(
        patchsize.get(holds, size) // (factor if holds == Axis.GROUND else 1)
        for size, holds in zip(shape, axes, strict=True)
    )


def read_patch_shapes(
    build: DatasetBuild,
    sizes: Mapping[str, Mapping[str, int]],
    pool: Mapping[str, int],
) -> dict[str, tuple[int, ...]]:
    """Return the shape of one patch of each instrument as the model reads it.

    Args:
        build: The published build the instruments are read from.
        sizes: How far a patch of each sensor runs along each axis it is cut on.
        pool: How many ground samples of a patch each instrument averages into one.

    Returns:
        shapes: The shape of one patch of each instrument, keyed as ODE names it.
    """
    rows = build.read_row_by_instrument()
    return {
        name: patch_lengths(
            rows[name].shape,
            rows[name].axes,
            size | read_band_patchsize(name),
            pool.get(name, 1),
        )
        for name, size in sizes.items()
    }


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
        arrays: The patches under "values", "measured" and "position".
    """
    measured_shape = tuple(
        held if holds == Axis.GROUND else 1
        for held, holds in zip(shape, axes, strict=True)
    )
    return {
        "values": np.array([one.values for one in patches], np.float32).reshape(
            -1, *shape
        ),
        "measured": np.array([one.measured for one in patches], bool).reshape(
            -1, *measured_shape
        ),
        "position": np.array(
            [[one.east_m, one.north_m, one.delay] for one in patches],
            np.float32,
        ).reshape(-1, 3),
    }


def cropped_patch_arrays(
    sample: dict[str, dict[str, np.ndarray]],
    budget: Mapping[str, int],
    draw: random.Random,
) -> dict[str, dict[str, np.ndarray]]:
    """Return a tile's patch arrays, each instrument cut to the ones nearest a centre.

    Args:
        sample: Each instrument's patch arrays over the tile.
        budget: How many patches of each instrument are kept.
        draw: What picks the centre.

    Returns:
        sample: Each instrument's kept patch arrays, nearest the centre first.
    """
    grounds = np.concatenate([arrays["position"][:, :2] for arrays in sample.values()])
    # A tile holding no patch has nothing to cut
    if not len(grounds):
        return sample
    # One patch of any instrument is the centre every instrument is cut around
    centre = grounds[draw.randrange(len(grounds))]
    cropped = {}
    for name, arrays in sample.items():
        apart = np.linalg.norm(arrays["position"][:, :2] - centre, axis=1)
        kept = np.argsort(apart, kind="stable")[: budget[name]]
        cropped[name] = {key: array[kept] for key, array in arrays.items()}
    return cropped
