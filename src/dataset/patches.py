"""Cutting one observation into the patches a transformer reads it as."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence

import numpy as np
from building.common.layout import WAVELENGTH
from building.metadata.observation import ObservationMetadata
from building.preprocessing.common.store import DEGREES, EAST, NORTH
from shared.maths import geodesy

from dataset.models.observation import Observation
from dataset.models.patch import Patch
from dataset.store import DatasetBuild


def load_patches(
    observations: Iterable[ObservationMetadata],
    build: DatasetBuild,
    patchsize: dict[str, int],
) -> Iterator[Patch]:
    """Yield every patch of every observation of one feature.

    Args:
        observations: The feature's own index rows, every observation the build
            holds of it.
        build: The published build the observations are read from.
        patchsize: How far a patch runs along every axis it is cut on, by
            instrument, and under "default" for every instrument unnamed.

    Yields:
        patch: Every patch of every observation, one at a time, so only the
            observation being cut is held in memory. A feature runs to over a
            million.
    """
    for record in observations:
        cut_by = patchsize.get(record.instrument, patchsize["default"])
        yield from cut_patches(build.read_observation(record.path), record, cut_by)


def cut_patches(
    observation: Observation, record: ObservationMetadata, patchsize: int
) -> Iterator[Patch]:
    """Yield every patch of one observation.

    Args:
        observation: The observation to cut, whose axes say which of them are
            tiled.
        record: What the index says the observation is, which carries the ground
            it spans and when it was taken.
        patchsize: How far a patch of this instrument runs along every axis it
            is cut on.

    Yields:
        patch: Every whole patch of the observation, in the order its axes run,
            each carrying which of its samples were measured. One CTX scan holds
            tens of thousands, so they are yielded one at a time and never
            gathered: a feature runs to over a million.
    """
    lengths = patch_lengths(observation.values.shape, observation.axes, patchsize)
    counts = patch_counts(observation.values.shape, observation.axes, patchsize)
    for one in range(math.prod(counts)):
        origin = tuple(
            int(at) * length
            for at, length in zip(np.unravel_index(one, counts), lengths, strict=True)
        )
        window = tuple(
            slice(start, start + length)
            for start, length in zip(origin, lengths, strict=True)
        )
        valid = observation.measured[tuple(window[at] for at in observation.ground)]
        north_m, east_m = patch_position(observation, window)
        yield Patch(
            instrument=observation.instrument,
            identifier=observation.identifier,
            # Copied, so one patch does not hold the whole observation behind it.
            values=observation.values[window].copy(),
            valid=valid.copy(),
            axes=observation.axes,
            origin=origin,
            ground_sample_m=record.ground_sample_m,
            beside=_beside(observation, window),
            north_m=north_m,
            east_m=east_m,
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
        patchsize: How far a patch of this instrument runs along an axis it is
            cut on.

    Returns:
        lengths: One length per axis, in the axes' own order. A wavelength axis
            is kept whole, which is how its bands become the channels of every
            patch cut from the observation.
    """
    return tuple(
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
        patchsize: How far a patch of this instrument runs along an axis it is
            cut on.

    Returns:
        counts: How many whole patches each axis holds. What is left of an axis
            after the last whole one is dropped rather than padded, so an axis
            shorter than one patch holds none and the observation holds none.
    """
    return tuple(
        size // length
        for size, length in zip(
            shape, patch_lengths(shape, axes, patchsize), strict=True
        )
    )


def patch_position(
    observation: Observation, window: Sequence[slice]
) -> tuple[float, float]:
    """Return how far the centre of one patch sits from the feature centre.

    Args:
        observation: The observation it was cut from, which carries the offset
            every sample of it stands at and the unit those are held in.
        window: What the patch keeps of each axis of the values, in the axes'
            own order.

    Returns:
        north_m: How far north of the feature centre the patch centre sits, in
            metres.
        east_m: How far east of it, in metres.
    """
    held = dict(zip(observation.dims[observation.measurement], window, strict=True))

    def middle(name: str) -> float:
        """Return one offset array where the middle of the patch falls on it."""
        taken = tuple(
            (held[one].start + held[one].stop) // 2 for one in observation.dims[name]
        )
        return float(getattr(observation, name)[taken])

    down, across = middle(NORTH), middle(EAST)
    # A projected observation stands in metres already, every other in degrees.
    if observation.position_units != DEGREES:
        return down, across
    return (
        geodesy.northward_m(down),
        geodesy.eastward_m(across, observation.centre_lat),
    )


def _beside(
    observation: Observation, window: tuple[slice, ...]
) -> dict[str, np.ndarray]:
    """Return what the instrument stores beside its values, cut to one patch.

    Args:
        observation: The observation it was cut from, which names the axes of
            every array it stores.
        window: What the patch keeps of each axis of the values.

    Returns:
        beside: Each of those arrays, keyed as it is written, cut along every
            axis it shares with the values and kept whole along the rest.
            Copied, for the same reason the values are. Empty for an instrument
            that stores none.
    """
    taken = dict(zip(observation.dims[observation.measurement], window, strict=True))
    return {
        name: held[
            tuple(taken.get(one, slice(None)) for one in observation.dims[name])
        ].copy()
        for name, held in observation.beside.items()
    }
