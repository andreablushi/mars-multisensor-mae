"""Cutting one observation into the patches a transformer reads it as."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from itertools import chain
from typing import TYPE_CHECKING

import numpy as np
from building.common.layout import ELEVATION, GROUND, WAVELENGTH
from building.metadata.observation import ObservationMetadata

from dataset.models.observation import Observation
from dataset.models.patch import Patch

if TYPE_CHECKING:
    from dataset.store import DatasetBuild

HEIGHTS = "elevation"


def patch_sizes(
    instruments: Sequence[str], patchsize: Mapping[str, int]
) -> dict[str, int]:
    """Return how far a patch of each instrument runs along an axis it is cut on.

    Args:
        instruments: The instruments the model reads, as ODE names them.
        patchsize: How far a patch runs, by instrument, keyed as ODE names it.

    Returns:
        sizes: One length per instrument, keyed as ODE names it.

    Raises:
        KeyError: When an instrument the model reads has no length written for it.
    """
    return {name: patchsize[name] for name in instruments}


def read_feature_patches(
    rows: Mapping[str, Sequence[ObservationMetadata]],
    build: DatasetBuild,
    sizes: Mapping[str, int],
    heights: np.ndarray,
    budget: int,
    overlap: float,
    draw: random.Random,
) -> dict[str, list[Patch]]:
    """Return a bounded draw of every instrument's patches over one feature.

    Args:
        rows: The feature's index rows of each instrument, keyed as ODE names it.
        build: The published build the observations are read from.
        sizes: How far a patch of each instrument runs along an axis it is cut on.
        heights: Where the ground stands over the feature. (N, 3)
        budget: How many patches of each instrument the draw runs to at most.
        overlap: The share of every other instrument's patches that must reach
            ground the anchor's patches reach.
        draw: What picks each observation and the patches taken from it.

    Returns:
        read: At most `budget` patches of one observation of each instrument,
            keyed as ODE names it, and none that measured nothing. The
            instrument holding the fewest patches anchors the draw and every
            other meets it over `overlap` of its own, so a cross-modal
            reconstruction is asked of patches seeing one place rather than
            one feature.
    """
    records, observations, boxes = {}, {}, {}
    for name, size in sizes.items():
        held = rows.get(name, ())
        whole = [math.prod(patch_counts(one.shape, one.axes, size)) for one in held]
        if not sum(whole):
            continue
        # One observation alone, since reading it is what reading any patch of it
        # costs, weighted so every patch of the feature is as likely as any other.
        (picked,) = draw.choices(range(len(held)), weights=whole)
        records[name] = held[picked]
        observations[name] = build.read_observation(records[name].path)
        boxes[name] = patch_boxes(observations[name], size)
    read: dict[str, list[Patch]] = {name: [] for name in sizes}
    if not records:
        return read
    # The scarcest instrument anchors, being the one every other can always meet.
    anchor = min(boxes, key=lambda name: len(boxes[name][0]))
    covered: tuple[np.ndarray, np.ndarray] | None = None
    for name in (anchor, *(one for one in records if one != anchor)):
        low, high = boxes[name]
        if covered is None:
            chosen = draw.sample(range(len(low)), min(budget, len(low)))
        else:
            against_low, against_high = covered
            reaching = (
                (
                    (low[:, None] <= against_high[None])
                    & (high[:, None] >= against_low[None])
                )
                .all(axis=2)
                .any(axis=1)
            )  # (T,)
            near = np.flatnonzero(reaching).tolist()
            wanted = min(round(overlap * budget), len(near), budget)
            chosen = draw.sample(near, wanted)
            taken = set(chosen)
            spare = np.flatnonzero(~reaching).tolist() + [
                one for one in near if one not in taken
            ]
            chosen += draw.sample(spare, min(budget - wanted, len(spare)))
        cut = (
            cut_patch(observations[name], records[name], index, sizes[name], heights)
            for index in chosen
        )
        read[name] = [patch for patch in cut if patch.valid.any()]
        if covered is None:
            covered = (low[chosen], high[chosen])
    return read


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
    ground_shape = tuple(
        length if holds == GROUND else 1
        for length, holds in zip(lengths, axes, strict=True)
    )
    valid = observation.measured[taken].reshape(ground_shape).copy()
    if WAVELENGTH in axes and record.band_valid_count is not None:
        # A band the observation never measured was filled, so it measures nothing.
        band_shape = tuple(-1 if holds == WAVELENGTH else 1 for holds in axes)
        measured = np.asarray(record.band_valid_count) > 0
        valid = valid & measured.reshape(band_shape)
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
        valid=valid,
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


def patch_boxes(
    observation: Observation, patchsize: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the ground each patch of one observation reaches, by its own index.

    The positions are read at the ends of each patch alone rather than at every
    sample of it, which a whole scan holds more of than the scan itself. The
    projection runs smoothly over one patch, so its corners are where it
    reaches furthest.

    Args:
        observation: The observation, read whole.
        patchsize: How far a patch of this instrument runs along an axis it is
            cut on.

    Returns:
        low: The least east and the least north each whole patch reaches, in
            metres of the feature's own frame, in the order `cut_patch` counts
            them. (T, 2)
        high: The greatest each reaches, holding the same. (T, 2)
    """
    shape, axes = observation.values.shape, observation.axes
    counts = patch_counts(shape, axes, patchsize)
    lengths = patch_lengths(shape, axes, patchsize)
    blocks = tuple(counts[at] for at in observation.ground)
    spans = tuple(lengths[at] for at in observation.ground)
    edges = [
        np.stack(
            [np.arange(count) * span, (np.arange(count) + 1) * span - 1], axis=1
        ).ravel()
        for count, span in zip(blocks, spans, strict=True)
    ]
    # A separable position holds one ground axis each and is crossed when it is read.
    taken = tuple(edges) if observation.described["separable"] else np.ix_(*edges)
    folded = tuple(chain.from_iterable((count, 2) for count in blocks))
    over = tuple(range(1, 2 * len(blocks), 2))
    corners = [
        one.reshape(folded) for one in reversed(observation.ground_metres(taken))
    ]  # east then north
    low = np.stack([one.min(over) for one in corners], axis=-1)  # (*blocks, 2)
    high = np.stack([one.max(over) for one in corners], axis=-1)  # (*blocks, 2)
    at = np.indices(counts).reshape(len(counts), -1)  # (A, T)
    held = tuple(at[one] for one in observation.ground)
    return low[held], high[held]  # (T, 2)


def read_patch_layout(
    build: DatasetBuild, sizes: Mapping[str, int]
) -> tuple[dict[str, tuple[int, ...]], dict[str, float]]:
    """Return the shape of one patch of each instrument, and how far two sit apart.

    Args:
        build: The published build the instruments are read from.
        sizes: How far a patch of each instrument runs along an axis it is cut
            on, keyed as ODE names it.

    Returns:
        shapes: The shape of one patch of each instrument, keyed as ODE names it.
        strides: How far apart two neighbouring patch centres of each
            instrument sit, in metres, which sets the shortest period its
            positions are read at.
    """
    rows = build.read_row_by_instrument()
    ground = build.read_ground_sample_by_instrument()
    return (
        {
            name: patch_lengths(rows[name].shape, rows[name].axes, size)
            for name, size in sizes.items()
        },
        {name: size * ground[name] for name, size in sizes.items()},
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
