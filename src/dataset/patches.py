"""Cutting one observation into the patches a transformer reads it as."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from building.common.layout import DELAY, GROUND, WAVELENGTH
from building.configs import sharad
from building.metadata.observation import ObservationMetadata
from shared.maths import physics

from dataset.models.observation import Observation
from dataset.models.patch import Patch

if TYPE_CHECKING:
    from dataset.store import DatasetBuild

METRES_PER_ROW = physics.SPEED_OF_LIGHT_M_S * sharad.DELAY_INTERVAL_S / 2.0


def delay_height_m(rows: np.ndarray) -> np.ndarray:
    """Return how high above the areoid each delay row of a radargram stands.

    Args:
        rows: The rows, counted from the top of the window as the radargram is.

    Returns:
        height_m: The metres above the areoid each of them sounds, the row the
            areoid itself lands on standing at nothing.
    """
    return (sharad.AREOID_ROW - np.asarray(rows, np.float64)) * METRES_PER_ROW


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
    heights: np.ndarray,
) -> dict[str, list[Patch]]:
    """Return every patch of every observation one tile holds, by instrument.

    Args:
        rows: The tile's index rows of each instrument, keyed as ODE names it.
        build: The published build the observations are read from.
        sizes: How far a patch of each instrument runs along each axis it is cut on.
        heights: Where the ground stands over the tile. (N, 3)

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
                patch = cut_patch(observation, record, at, lengths, counts, heights)
                if patch.valid.any():
                    held.append(patch)
        read[name] = held
    return read


def cut_patch(
    observation: Observation,
    record: ObservationMetadata,
    index: int,
    lengths: tuple[int, ...],
    counts: tuple[int, ...],
    heights: np.ndarray,
) -> Patch:
    """Return one whole patch of one observation.

    Args:
        observation: The observation, read whole.
        record: Its index row, which carries what a patch says about the whole.
        index: Which patch, counting whole ones as the axes run, the last fastest.
        lengths: How far one patch of it runs along each axis.
        counts: How many whole patches each of those axes holds.
        heights: Where the ground stands over the tile. (N, 3)

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
    north, east = observation.distance_centre_m(taken)
    north_m, east_m = float(np.mean(north)), float(np.mean(east))
    ground_shape = tuple(
        length if holds == GROUND else 1
        for length, holds in zip(lengths, axes, strict=True)
    )
    valid = observation.measured[taken].reshape(ground_shape).copy()
    if record.band_valid_count is not None:
        # A band the observation never measured was filled, so it measures nothing.
        band_shape = tuple(-1 if holds == WAVELENGTH else 1 for holds in axes)
        measured = np.asarray(record.band_valid_count) > 0
        valid = valid & measured.reshape(band_shape)
    if DELAY in axes:
        # A sounder reads its height off the delay, the same frame the heights are in.
        at = axes.index(DELAY)
        stood = delay_height_m(np.arange(origin[at], origin[at] + lengths[at]))
        height_m = float(stood.mean())
        height_span_m = float(np.ptp(stood))
    else:
        height_span_m = 0.0
        nearest = np.argmin(
            (heights[:, 0] - north_m) ** 2 + (heights[:, 1] - east_m) ** 2
        )
        height_m = float(heights[nearest, 2])
    return Patch(
        instrument=observation.instrument,
        identifier=observation.identifier,
        values=observation.values[window].copy(),
        valid=valid,
        axes=axes,
        origin=origin,
        north_m=north_m,
        east_m=east_m,
        height_m=height_m,
        north_span_m=float(np.ptp(north)),
        east_span_m=float(np.ptp(east)),
        height_span_m=height_span_m,
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
    build: DatasetBuild, sizes: Mapping[str, Mapping[str, int]]
) -> tuple[dict[str, tuple[int, ...]], dict[str, float]]:
    """Return the shape of one patch of each instrument, and how far two sit apart.

    Args:
        build: The published build the instruments are read from.
        sizes: How far a patch of each sensor runs along each axis it is cut on.

    Returns:
        shapes: The shape of one patch of each instrument, keyed as ODE names it.
        strides: How far apart two neighbouring patch centres of each sensor sit.
    """
    rows = build.read_row_by_instrument()
    ground = build.read_ground_sample_by_instrument()
    return (
        {
            name: patch_lengths(rows[name].shape, rows[name].axes, size)
            for name, size in sizes.items()
        },
        {name: size[GROUND] * ground[name] for name, size in sizes.items()},
    )
