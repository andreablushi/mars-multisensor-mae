"""Cutting one observation into the patches a transformer reads it as."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence

import numpy as np
from building.common.layout import WAVELENGTH
from building.metadata.observation import ObservationMetadata
from building.preprocessing.common.store import EAST, NORTH

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
        patch: Every whole patch of every observation, in the order its axes
            run, each carrying which of its samples were measured. They are
            yielded one at a time and never gathered, one observation held at a
            time: a CTX scan holds tens of thousands and a feature runs to over
            a million.
    """
    for record in observations:
        observation = build.read_observation(record.path)
        cut_by = patchsize.get(record.instrument, patchsize["default"])
        counts = patch_counts(observation.values.shape, observation.axes, cut_by)
        for one in range(math.prod(counts)):
            yield cut_patch(observation, record, one, cut_by)


def cut_patch(
    observation: Observation, record: ObservationMetadata, index: int, patchsize: int
) -> Patch:
    """Return one whole patch of one observation.

    Args:
        observation: The observation, read whole.
        record: Its index row, which carries what a patch says about the whole.
        index: Which patch, counting the whole ones in the order the axes run,
            the last axis fastest.
        patchsize: How far a patch of this instrument runs along an axis it is
            cut on.

    Returns:
        patch: The patch, carrying which of its samples were measured. Copied,
            so it does not hold the observation behind it.
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
    valid = observation.measured[tuple(window[at] for at in observation.ground)]
    north_m, east_m = patch_position(observation, window)
    return Patch(
        instrument=observation.instrument,
        identifier=observation.identifier,
        values=observation.values[window].copy(),
        valid=valid.copy(),
        axes=axes,
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
        observation: The observation it was cut from, which carries the ground
            metres every sample of it stands from the feature centre.
        window: What the patch keeps of each axis of the values, in the axes'
            own order.

    Returns:
        north_m: How far north of the feature centre the patch centre sits, in
            metres.
        east_m: How far east of it, in metres.
    """
    held = dict(zip(observation.dims[observation.measurement], window, strict=True))

    def middle(name: str) -> float:
        """Return where one offset array has the patch, over the samples it keeps."""
        taken = tuple(held[one] for one in observation.dims[name])
        return float(np.mean(getattr(observation, name)[taken]))

    return middle(NORTH), middle(EAST)


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
