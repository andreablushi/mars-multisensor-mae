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


def draw_feature_patches(
    rows: Mapping[str, Sequence[ObservationMetadata]],
    build: DatasetBuild,
    sizes: Mapping[str, int],
    wavelengths: Mapping[str, Sequence[float]],
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
        wavelengths: What each band of each spectral sensor is centred on, in nm.
        heights: Where the ground stands over the feature. (N, 3)
        budget: How many patches of each instrument the draw runs to at most.
        overlap: The share of each other sensor's patches over the anchor's ground.
        draw: What picks each observation and the patches taken from it.

    Returns:
        drawn: At most `budget` patches of one observation of each sensor, none empty.
    """
    # Initialize storage for records, loaded data, and patch bounding boxes
    records, observations, boxes = {}, {}, {}
    for name, size in sizes.items():
        held = rows.get(name, ())
        whole = [math.prod(patch_counts(one.shape, one.axes, size)) for one in held]
        if not sum(whole):
            continue
        # Pick one observation randomly, weighted by its total possible patches
        (picked,) = draw.choices(range(len(held)), weights=whole)
        records[name] = held[picked]
        observations[name] = build.read_observation(records[name].path)
        boxes[name] = patch_coordinates(observations[name], size)
    drawn: dict[str, list[Patch]] = {name: [] for name in sizes}
    if not records:
        return drawn
    # Identify the anchor instrument (fewest total patches)
    anchor = min(boxes, key=lambda name: len(boxes[name][0]))
    covered: tuple[np.ndarray, np.ndarray] | None = None
    # Process the anchor first, then the remaining instruments
    for name in (anchor, *(one for one in records if one != anchor)):
        low, high = boxes[name]
        if covered is None:
            # Anchor case: randomly sample up to the budget limit
            chosen = draw.sample(range(len(low)), min(budget, len(low)))
        else:
            # Other instruments: find patches that overlap with the anchor's area
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
            # Sample overlapping patches, then fill remaining budget with spares
            chosen = draw.sample(near, wanted)
            taken = set(chosen)
            spare = np.flatnonzero(~reaching).tolist() + [
                one for one in near if one not in taken
            ]
            chosen += draw.sample(spare, min(budget - wanted, len(spare)))
        # Cut out the valid patches and store them
        cut = (
            cut_patch(
                observations[name],
                records[name],
                index,
                sizes[name],
                heights,
                wavelengths.get(name, ()),
            )
            for index in chosen
        )
        drawn[name] = [patch for patch in cut if patch.valid.any()]
        # Save anchor coverage bounds on the first iteration
        if covered is None:
            covered = (low[chosen], high[chosen])
    return drawn


def read_feature_patches(
    rows: Mapping[str, Sequence[ObservationMetadata]],
    build: DatasetBuild,
    sizes: Mapping[str, int],
    wavelengths: Mapping[str, Sequence[float]],
    heights: np.ndarray,
    ceiling: int,
) -> dict[str, list[Patch]]:
    """Return every patch of every observation of one feature, bounded in count.

    Args:
        rows: The feature's index rows of each instrument, keyed as ODE names it.
        build: The published build the observations are read from.
        sizes: How far a patch of each instrument runs along an axis it is cut on.
        wavelengths: What each band of each spectral sensor is centred on, in nm.
        heights: Where the ground stands over the feature. (N, 3)
        ceiling: How many patches of each instrument one read hands back at most.

    Returns:
        read: The patches of each instrument, none empty, keyed as ODE names it.
    """
    read: dict[str, list[Patch]] = {}
    for name, size in sizes.items():
        planned = [
            (record, at)
            for record in rows.get(name, ())
            for at in range(math.prod(patch_counts(record.shape, record.axes, size)))
        ]
        # Over the ceiling the patches are thinned evenly, so a read still spans it.
        step = math.ceil(len(planned) / ceiling) or 1
        held: list[Patch] = []
        opened: tuple[str, Observation] | None = None
        # The patches of one observation run together, so it is read once for all.
        for record, at in planned[::step]:
            if opened is None or opened[0] != record.path:
                opened = (record.path, build.read_observation(record.path))
            patch = cut_patch(
                opened[1], record, at, size, heights, wavelengths.get(name, ())
            )
            if patch.valid.any():
                held.append(patch)
        read[name] = held
    return read


def cut_patch(
    observation: Observation,
    record: ObservationMetadata,
    index: int,
    patchsize: int,
    heights: np.ndarray,
    wavelengths: Sequence[float],
) -> Patch:
    """Return one whole patch of one observation.

    Args:
        observation: The observation, read whole.
        record: Its index row, which carries what a patch says about the whole.
        index: Which patch, counting whole ones as the axes run, the last fastest.
        patchsize: How far a patch of this instrument runs along an axis it is cut on.
        heights: Where the ground stands over the feature. (N, 3)
        wavelengths: What each band is centred on, in nm, empty for every other.

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
        # A sounder reads one delay at one height, which tells its channels apart.
        channels = np.asarray(beside[HEIGHTS], np.float32)
        height_m = float(np.mean(beside[HEIGHTS]))
        height_span_m = float(np.ptp(beside[HEIGHTS]))
    else:
        channels = np.asarray(wavelengths or [0.0], np.float32)
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
        channels=channels,
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
    at = [at for at, holds in enumerate(axes) if holds != GROUND]
    return at[0] if at else None


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


def patch_coordinates(
    observation: Observation, patchsize: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the ground each patch of one observation reaches, by its own index.

    Args:
        observation: The observation, read whole.
        patchsize: How far a patch of this instrument runs along an axis it is cut on.

    Returns:
        low: The least east and north each whole patch reaches, in metres. (T, 2)
        high: The greatest each reaches, holding the same. (T, 2)
    """
    shape, axes = observation.values.shape, observation.axes
    counts = patch_counts(shape, axes, patchsize)
    lengths = patch_lengths(shape, axes, patchsize)
    blocks = tuple(counts[at] for at in observation.ground_axes)
    spans = tuple(lengths[at] for at in observation.ground_axes)
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
    held = tuple(at[one] for one in observation.ground_axes)
    return low[held], high[held]  # (T, 2)


def read_patch_layout(
    build: DatasetBuild, sizes: Mapping[str, int]
) -> tuple[dict[str, tuple[int, ...]], dict[str, float]]:
    """Return the shape of one patch of each instrument, and how far two sit apart.

    Args:
        build: The published build the instruments are read from.
        sizes: How far a patch of each sensor runs along an axis it is cut on.

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
        beside: Each of those arrays, cut along every axis it shares with values.
    """
    taken = dict(zip(observation.dims[observation.measurement], window, strict=True))
    return {
        name: held[
            tuple(taken.get(one, slice(None)) for one in observation.dims[name])
        ].copy()
        for name, held in observation.beside.items()
    }
