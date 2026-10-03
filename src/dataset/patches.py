"""Cutting one observation into the patches a transformer reads it as."""

from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from building.common.layout import Axis
from building.metadata.observation import ObservationMetadata
from building.models.instrument import INSTRUMENTS
from building.preprocessing.common import geometry, relative_positioning
from building.preprocessing.common.models.position import Position
from building.preprocessing.common.store import STORED
from common.maths.geodesy import SPHEROID, PolarGrid
from common.models.tile import Tile

from dataset.models.observation import Observation
from dataset.models.patch import Patch

if TYPE_CHECKING:
    from dataset.store import DatasetBuild


def metres_from_centre(
    observation: Observation, taken: tuple[slice, ...] = ()
) -> tuple[np.ndarray, np.ndarray]:
    """Return how far north and east of its tile centre every sample one cut keeps sits.

    Args:
        observation: The stored observation the cut is taken of.
        taken: What the cut keeps of each ground axis, empty for every sample.

    Returns:
        north: The ground metres north of the centre, one per sample kept.
        east: The ground metres east of it, in the same frame.
    """
    described = observation.described
    separable = described["separable"]
    north, east = observation.north, observation.east
    if taken:
        north, east = (
            (north[taken[0]], east[taken[1]])
            if separable
            else (north[taken], east[taken])
        )
    grid = described["polar"]
    position = Position(
        north, east, separable, None if grid is None else PolarGrid(*grid)
    )
    frame = Tile(described["band"], described["column"], **described["box"])
    northing = np.empty(position.sizes, dtype=STORED)
    easting = np.empty(position.sizes, dtype=STORED)
    for block in geometry.line_blocks(position.sizes):
        lon, lat = relative_positioning.position_degrees(position, frame, block)
        lon, lat = np.broadcast_arrays(
            np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
        )
        # The geodesic each sample stands at, on the spheroid rather than a sphere.
        azimuth, _, span = SPHEROID.inv(
            np.full(lon.shape, frame.centre_lon),
            np.full(lon.shape, frame.centre_lat),
            lon,
            lat,
        )
        bearing = np.radians(azimuth)
        easting[block] = span * np.sin(bearing)
        northing[block] = span * np.cos(bearing)
    return northing, easting


def read_surface_delays(
    build: DatasetBuild, observations: Sequence[ObservationMetadata]
) -> np.ndarray:
    """Return the row every sample of the delay instrument sounds at, over a tile.

    Args:
        build: The published build the observations are read from.
        observations: The tile's index rows of the instrument carrying the delay.

    Returns:
        delays: One row per sample: north, east and the delay row. (N, 3)
    """
    placed = []
    for record in observations:
        # Loads specifically the delay plane, which is beside the measured values.
        observation = build.read_observation(record.path, ("delay",))
        measured = observation.measured
        north, east = metres_from_centre(observation)
        rows = observation.beside["delay"]
        # Unmeasured samples have no delay row.
        placed.append(
            np.stack([north[measured], east[measured], rows[measured]], axis=1)
        )  # (n, 3)
    return np.concatenate(placed).astype(np.float64)  # (N, 3)


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
            counts = tuple(
                length // patch
                for length, patch in zip(record.shape, lengths, strict=True)
            )
            observation = build.read_observation(record.path)
            for at in range(math.prod(counts)):
                patch = cut_patch(observation, record, at, lengths, counts, delays)
                if patch.valid.any():
                    if name in pool:
                        patch = downsampled_patch(patch, pool[name])
                    held.append(patch)
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
    north, east = metres_from_centre(
        observation, corners if observation.described["separable"] else taken
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
        delay = origin[at] + lengths[at] / 2
        delay_span = float(lengths[at])
    else:
        # A patch on the ground sounds at one row, the one its surface echo lands on.
        delay_span = 0.0
        nearest = np.argmin(
            (delays[:, 0] - north_m) ** 2 + (delays[:, 1] - east_m) ** 2
        )
        delay = float(delays[nearest, 2]) + 0.5
    side = lengths[observation.ground_axes[0]]
    footprint = side / max(side - 1, 1)
    spans = (np.ptp(north), np.ptp(east))
    if north.ndim == 2:
        down = [np.mean(one[-1] - one[0]) for one in (north, east)]
        across = [np.mean(one[:, -1] - one[:, 0]) for one in (north, east)]
        if abs(down[1] * across[0]) > abs(down[0] * across[1]):
            down, across = across, down
        spans = (abs(down[0]), abs(across[1]))
    return Patch(
        values=observation.values[window].copy(),
        valid=valid,
        axes=axes,
        north_m=north_m,
        east_m=east_m,
        delay=delay,
        north_span_m=float(spans[0]) * footprint,
        east_span_m=float(spans[1]) * footprint,
        delay_span=delay_span,
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
    valid = patch.valid.reshape(split(patch.valid.shape))
    weight = np.broadcast_to(valid, split(patch.values.shape)).astype(np.float32)
    values = patch.values.reshape(weight.shape)
    return dataclasses.replace(
        patch,
        values=values.sum(axis=pooled) / np.maximum(weight.sum(axis=pooled), 1.0),
        valid=valid.any(axis=pooled),
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


def read_patch_layout(
    build: DatasetBuild,
    sizes: Mapping[str, Mapping[str, int]],
    pool: Mapping[str, int],
) -> tuple[
    dict[str, tuple[int, ...]], dict[str, float], dict[str, tuple[float, ...] | None]
]:
    """Return each instrument's patch shape, patch spacing and channel wavelengths.

    Args:
        build: The published build the instruments are read from.
        sizes: How far a patch of each sensor runs along each axis it is cut on.
        pool: How many ground samples of a patch each instrument averages into one.

    Returns:
        shapes: The shape of one patch of each instrument, keyed as ODE names it.
        strides: How far apart two neighbouring patch centres of each sensor sit.
        centres_nm: The wavelength of each sensor's channels, in nm, or None.
    """
    rows = build.read_row_by_instrument()
    spacing = defaultdict(list)
    for held in build.read_observation_metadata().values():
        for name, records in held.items():
            spacing[name].extend(min(one.sample_spacing_m) for one in records)
    resolution = {name: float(np.median(held)) for name, held in spacing.items()}
    return (
        {
            name: patch_lengths(
                rows[name].shape, rows[name].axes, size, pool.get(name, 1)
            )
            for name, size in sizes.items()
        },
        {name: size[Axis.GROUND] * resolution[name] for name, size in sizes.items()},
        {name: INSTRUMENTS[name].layout.band_centres_nm for name in sizes},
    )
