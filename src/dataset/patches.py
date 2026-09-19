"""Cutting one observation into the patches a transformer reads it as."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from building.common.layout import DELAY, GROUND
from building.metadata.observation import ObservationMetadata
from torch import Tensor

from dataset.models.observation import Observation
from dataset.models.patch import Patch

if TYPE_CHECKING:
    from dataset.store import DatasetBuild

HEIGHTS = "elevation"


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
            counts = patch_counts(record.shape, record.axes, size)
            if not math.prod(counts):
                continue
            observation = build.read_observation(record.path)
            for at in range(math.prod(counts)):
                patch = cut_patch(observation, record, at, size, heights)
                if patch.valid.any():
                    held.append(patch)
        read[name] = held
    return read


def cut_patch(
    observation: Observation,
    record: ObservationMetadata,
    index: int,
    patchsize: Mapping[str, int],
    heights: np.ndarray,
) -> Patch:
    """Return one whole patch of one observation.

    Args:
        observation: The observation, read whole.
        record: Its index row, which carries what a patch says about the whole.
        index: Which patch, counting whole ones as the axes run, the last fastest.
        patchsize: How far a patch of this instrument runs along each axis it is cut on.
        heights: Where the ground stands over the tile. (N, 3)

    Returns:
        patch: The patch, copied, with what it measured and where it reaches.
    """
    shape, axes = observation.values.shape, observation.axes
    lengths = patch_lengths(shape, axes, patchsize)
    counts = patch_counts(shape, axes, patchsize)
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
    by_dim = dict(zip(observation.dims[observation.measurement], window, strict=True))
    beside = {
        name: held[
            tuple(by_dim.get(one, slice(None)) for one in observation.dims[name])
        ].copy()
        for name, held in observation.beside.items()
    }
    if DELAY in axes:
        # A sounder reads one delay at one height, which tells its rows apart.
        elevations = beside[HEIGHTS]
        height_m = float(np.mean(elevations))
        height_span_m = float(np.ptp(elevations))
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
        beside=beside,
        north_m=north_m,
        east_m=east_m,
        height_m=height_m,
        north_span_m=float(np.ptp(north)),
        east_span_m=float(np.ptp(east)),
        height_span_m=height_span_m,
        t_start=record.t_start,
        t_end=record.t_end,
    )


def channel_axis(axes: Sequence[str]) -> int | None:
    """Return which axis of a patch its channels run along.

    Args:
        axes: What each axis of the instrument's values holds.

    Returns:
        at: The one axis that is not ground, or None where a patch is ground alone.
    """
    return next((at for at, holds in enumerate(axes) if holds != GROUND), None)


def normalize_patches(
    values: Tensor, valid: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return each patch centred and scaled by the measured samples of its own.

    Args:
        values: The patches, as the model was handed them. (B, K, *P)
        valid: Whether each sample is a measurement, broadcastable to them. (B, K, *P')

    Returns:
        target: The patches, each of zero mean and unit deviation. (B, K, *P)
        counted: Whether each sample is a measurement, spread over them. (B, K, *P)
        mean: What each patch was centred by, to undo it. (B, K, 1...)
        deviation: What each was scaled by, holding the same. (B, K, 1...)
    """
    counted = valid.to(values.dtype).expand_as(values)  # (B, K, *P)
    over = tuple(range(2, values.dim()))
    spread = (*values.shape[:2], *([1] * len(over)))
    samples = counted.sum(dim=over).clamp(min=1)  # (B, K)
    mean = ((values * counted).sum(dim=over) / samples).reshape(spread)  # (B, K, 1...)
    variance = ((values - mean) ** 2 * counted).sum(dim=over) / samples  # (B, K)
    deviation = (variance.reshape(spread) + 1e-6).sqrt()  # (B, K, 1...)
    return (values - mean) / deviation, counted, mean, deviation


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
