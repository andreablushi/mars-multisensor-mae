"""Cutting one observation into the patches a transformer reads it as."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from building.common.layout import ELEVATION, WAVELENGTH
from building.metadata.observation import ObservationMetadata

from dataset.models.observation import Observation
from dataset.models.patch import Patch

if TYPE_CHECKING:
    from dataset.store import DatasetBuild

HEIGHTS = "elevation"

DEFAULT_PATCHSIZE = "default"

# How many patches of one instrument one step carries, which the widest
# reconstruction settles: a decoder predicts a whole patch of its instrument for
# every instrument read, so this many CTX patches is what a 24Gi card holds.
PATCHES_PER_STEP = 512


def patch_sizes(
    instruments: Sequence[str], patchsize: Mapping[str, int]
) -> dict[str, int]:
    """Return how far a patch of each instrument runs along an axis it is cut on.

    Args:
        instruments: The instruments the model reads, as ODE names them.
        patchsize: How far a patch runs, by instrument, and under "default" for
            every instrument unnamed.

    Returns:
        sizes: One length per instrument, keyed as ODE names it.
    """
    return {
        name: patchsize.get(name, patchsize[DEFAULT_PATCHSIZE]) for name in instruments
    }


def read_patches(
    observations: Sequence[ObservationMetadata],
    build: DatasetBuild,
    patchsize: int,
    heights: np.ndarray,
    budget: int,
    draw: random.Random,
) -> list[Patch]:
    """Return a bounded draw of one instrument's patches over one feature.

    Args:
        observations: The feature's index rows of that one instrument.
        build: The published build the observations are read from.
        patchsize: How far a patch of that instrument runs along.
        heights: Where the ground stands over the feature. (N, 3)
        budget: How many patches the draw runs to at most.
        draw: What picks the observation and the patches taken from it.

    Returns:
        read: At most `budget` patches of one observation of the feature, drawn
            without replacement and none that measured nothing. The observation
            is drawn against how many whole patches each holds, so every patch
            of the feature stands the same chance of being read. Empty where
            the feature holds none.
    """
    whole = [
        math.prod(patch_counts(record.shape, record.axes, patchsize))
        for record in observations
    ]
    if not sum(whole):
        return []
    # One observation alone, since reading it is what reading any patch of it costs.
    (picked,) = draw.choices(range(len(observations)), weights=whole)
    record, held = observations[picked], whole[picked]
    observation = build.read_observation(record.path)
    read = (
        cut_patch(observation, record, index, patchsize, heights)
        for index in draw.sample(range(held), min(budget, held))
    )
    return [patch for patch in read if patch.valid.any()]


def cut_patch(
    observation: Observation,
    record: ObservationMetadata,
    index: int,
    patchsize: int,
    heights: np.ndarray,
) -> Patch:
    """Return one whole patch of one observation.

    Args:
        observation: The observation, read whole.
        record: Its index row, which carries what a patch says about the whole.
        index: Which patch, counting the whole ones in the order the axes run,
            the last axis fastest.
        patchsize: How far a patch of this instrument runs along an axis it is
            cut on.
        heights: Where the ground stands over the feature, as the elevation
            instrument measured it. (N, 3)

    Returns:
        patch: The patch, carrying which of its samples were measured, how high
            its centre stands, and how far it reaches in each direction, which
            is what tells a surface tile from a sounding column. Copied, so it
            does not hold the observation behind it.
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
    taken = tuple(window[at] for at in observation.ground)
    north, east = observation.ground_metres(taken)
    north_m, east_m = float(np.mean(north)), float(np.mean(east))
    beside = _beside(observation, window)
    if ELEVATION in axes:
        height_m = float(np.mean(beside[HEIGHTS]))
        height_span_m = float(np.ptp(beside[HEIGHTS]))
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
        valid=observation.measured[taken].copy(),
        axes=axes,
        origin=origin,
        ground_sample_m=record.ground_sample_m,
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


def patch_lengths(
    shape: Sequence[int], axes: Sequence[str], patchsize: int
) -> tuple[int, ...]:
    """Return how far one patch runs along each axis of an observation.

    Args:
        shape: How many samples each axis of the values holds.
        axes: What each of those axes holds, in that same order.
        patchsize: How far a patch of this instrument runs along.

    Returns:
        lengths: One length per axis, in the axes' own order.
    """
    return tuple(
        # The wavelength axis is not cut, so a patch runs the whole way along it.
        size if holds == WAVELENGTH else patchsize
        for size, holds in zip(shape, axes, strict=True)
    )


def patch_counts(
    shape: Sequence[int], axes: Sequence[str], patchsize: int
) -> tuple[int, ...]:
    """Return how many patches fit along each axis of an observation.

    Args:
        shape: How many samples each axis of the values holds.
        axes: What each of those axes holds, in that same order.
        patchsize: How far a patch of this instrument runs along.

    Returns:
        counts: How many whole patches each axis holds.
    """
    return tuple(
        size // length
        for size, length in zip(
            shape, patch_lengths(shape, axes, patchsize), strict=True
        )
    )


def _beside(
    observation: Observation, window: tuple[slice, ...]
) -> dict[str, np.ndarray]:
    """Return what the instrument stores beside its values, cut to one patch.

    Args:
        observation: The observation it was cut from.
        window: What the patch keeps of each axis of the values.

    Returns:
        beside: Each of those arrays, keyed as it is written, cut along every
            axis it shares with the values and kept whole along the rest.
    """
    taken = dict(zip(observation.dims[observation.measurement], window, strict=True))
    return {
        name: held[
            tuple(taken.get(one, slice(None)) for one in observation.dims[name])
        ].copy()
        for name, held in observation.beside.items()
    }
