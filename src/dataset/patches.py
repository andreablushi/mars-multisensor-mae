"""Cutting one observation into the patches a transformer reads it as."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from building.common.layout import Axis
from building.metadata.observation import ObservationMetadata

from dataset.models.observation import Observation
from dataset.models.patch import Patch
from dataset.per_instrument import pooled_patch, pooled_patch_shape

if TYPE_CHECKING:
    from dataset.store import DatasetBuild


def patch_sizes(
    instruments: Sequence[str], patchsize: Mapping[str, Mapping[str, int]]
) -> dict[str, Mapping[str, int]]:
    """Return how far a patch of each instrument runs along each axis it is cut on.

    Args:
        instruments: The instruments the model reads, as ODE names them.
        patchsize: How far a patch runs along an axis holding each thing, by
            instrument, keyed as ODE names it.

    Returns:
        sizes: One length per thing an axis holds, per instrument.

    Raises:
        KeyError: When an instrument the model reads has no length written for it.
    """
    return {name: patchsize[name] for name in instruments}


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
            lengths = patch_lengths(record.shape, record.axes, size)
            counts = patch_counts(record.shape, record.axes, size)
            if not math.prod(counts):
                continue
            observation = build.read_observation(record.path)
            for at in range(math.prod(counts)):
                patch = cut_patch(observation, record, at, lengths, counts, delays)
                if patch.valid.any():
                    held.append(pooled_patch(patch, pool))
        read[name] = held
    return read


def cut_patch(
    observation: Observation,
    record: ObservationMetadata,
    index: int,
    lengths: tuple[int, ...],
    counts: tuple[int, ...],
    delays: np.ndarray,
) -> Patch:
    """Return one whole patch of one observation.

    Args:
        observation: The observation, read whole.
        record: Its index row, which carries what a patch says about the whole.
        index: Which patch, counting whole ones as the axes run, the last fastest.
        lengths: How far one patch of it runs along each axis.
        counts: How many whole patches each of those axes holds.
        delays: Which delay row the ground sounds at, over the tile. (N, 3)

    Returns:
        patch: The patch, copied, with what it measured and where it reaches.
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
    north, east = observation.distance_centre_m(
        corners if observation.described["separable"] else taken
    )
    north_m, east_m = float(np.mean(north)), float(np.mean(east))
    ground_shape = tuple(
        length if holds == Axis.GROUND else 1
        for length, holds in zip(lengths, axes, strict=True)
    )
    valid = observation.measured[taken].reshape(ground_shape).copy()
    if record.band_valid_count is not None:
        # A band the observation never measured was filled, so it measures nothing.
        band_shape = tuple(-1 if holds == Axis.WAVELENGTH else 1 for holds in axes)
        measured = np.asarray(record.band_valid_count) > 0
        valid = valid & measured.reshape(band_shape)
    if Axis.DELAY in axes:
        # A sounder is placed by the rows it sounded, which is the patch's own cut.
        at = axes.index(Axis.DELAY)
        rows = np.arange(origin[at], origin[at] + lengths[at])
        delay = float(rows.mean())
        delay_span = float(np.ptp(rows))
    else:
        # A patch on the ground sounds at one row, the one its surface echo lands on.
        delay_span = 0.0
        nearest = np.argmin(
            (delays[:, 0] - north_m) ** 2 + (delays[:, 1] - east_m) ** 2
        )
        delay = float(delays[nearest, 2])
    return Patch(
        instrument=observation.instrument,
        identifier=observation.identifier,
        values=observation.values[window].copy(),
        valid=valid,
        axes=axes,
        origin=origin,
        north_m=north_m,
        east_m=east_m,
        delay=delay,
        north_span_m=float(np.ptp(north)),
        east_span_m=float(np.ptp(east)),
        delay_span=delay_span,
        t_start=record.t_start,
        t_end=record.t_end,
    )


def patch_lengths(
    shape: Sequence[int], axes: Sequence[str], patchsize: Mapping[str, int]
) -> tuple[int, ...]:
    """Return how far one patch runs along each axis of an observation.

    Args:
        shape: How many samples each axis of the values holds.
        axes: What each of those axes holds, in that same order.
        patchsize: How far a patch runs along an axis holding each thing.

    Returns:
        lengths: One length per axis, an axis the patchsize omits taken whole.
    """
    return tuple(
        patchsize.get(holds, size) for size, holds in zip(shape, axes, strict=True)
    )


def patch_counts(
    shape: Sequence[int], axes: Sequence[str], patchsize: Mapping[str, int]
) -> tuple[int, ...]:
    """Return how many patches fit along each axis of an observation.

    Args:
        shape: How many samples each axis of the values holds.
        axes: What each of those axes holds, in that same order.
        patchsize: How far a patch runs along an axis holding each thing.

    Returns:
        counts: How many whole patches each axis holds.
    """
    return tuple(
        size // length
        for size, length in zip(
            shape, patch_lengths(shape, axes, patchsize), strict=True
        )
    )


def read_patch_layout(
    build: DatasetBuild,
    sizes: Mapping[str, Mapping[str, int]],
    pool: Mapping[str, int],
) -> tuple[dict[str, tuple[int, ...]], dict[str, float]]:
    """Return the shape of one patch of each instrument, and how far two sit apart.

    Args:
        build: The published build the instruments are read from.
        sizes: How far a patch of each sensor runs along each axis it is cut on.
        pool: How many ground samples of a patch each instrument averages into one.

    Returns:
        shapes: The shape of one patch of each instrument, keyed as ODE names it.
        strides: How far apart two neighbouring patch centres of each sensor sit.
    """
    rows = build.read_row_by_instrument()
    ground = build.read_ground_sample_by_instrument()
    return (
        {
            name: pooled_patch_shape(
                patch_lengths(rows[name].shape, rows[name].axes, size),
                rows[name].axes,
                pool.get(name, 1),
            )
            for name, size in sizes.items()
        },
        {name: size[Axis.GROUND] * ground[name] for name, size in sizes.items()},
    )
