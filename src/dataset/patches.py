"""Cutting one observation into the patches a transformer reads it as."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from mars_multisensor_dataset.building.common.layout import Axis
from mars_multisensor_dataset.building.configs import sharad
from mars_multisensor_dataset.building.metadata.observation import ObservationMetadata

from dataset.models.observation import Observation
from dataset.models.patch import Patch

if TYPE_CHECKING:
    from dataset.store import DatasetBuild


def cut_tile_patches(
    rows: Mapping[str, Sequence[ObservationMetadata]],
    build: DatasetBuild,
    sizes: Mapping[str, Mapping[str, int]],
) -> dict[str, list[Patch]]:
    """Return every patch of every observation one tile holds, by instrument.

    Args:
        rows: The tile's index rows of each instrument, keyed as ODE names it.
        build: The published build the observations are read from.
        sizes: How far a patch of each instrument runs along each axis it is cut on.

    Returns:
        read: The patches of each instrument, none empty, keyed as ODE names it.
    """
    read: dict[str, list[Patch]] = {}
    for name, size in sizes.items():
        held: list[Patch] = []
        for record in rows[name]:
            observation = build.read_observation(record.path)
            angles = np.array(
                [getattr(record.acquisition, one) for one in build.acquisition[name]],
                np.float32,
            )
            shape = observation.values.shape
            lengths = patch_lengths(shape, record.axes, size)
            counts = tuple(
                length // patch for length, patch in zip(shape, lengths, strict=True)
            )
            if not math.prod(counts):
                continue
            ground = [
                at for at, holds in enumerate(record.axes) if holds == Axis.GROUND
            ]
            centres = observation.patch_metres_from_centre(
                [lengths[at] for at in ground], [counts[at] for at in ground]
            )
            for at in range(math.prod(counts)):
                patch = cut_patch(observation, at, lengths, counts, angles, centres)
                if patch.measured.any():
                    held.append(patch)
        read[name] = held
    return read


def cut_patch(
    observation: Observation,
    index: int,
    lengths: tuple[int, ...],
    counts: tuple[int, ...],
    acquisition: np.ndarray,
    centres: tuple[np.ndarray, np.ndarray],
) -> Patch:
    """Return one whole patch of one observation.

    Args:
        observation: The observation, read whole.
        index: Which patch, counting whole ones as the axes run, the last fastest.
        lengths: How far one patch of it runs along each axis.
        counts: How many whole patches each of those axes holds.
        acquisition: The observation's acquisition angles the model reads.
        centres: How far north and east of the tile centre every whole patch of it
            sits, one per patch along every ground axis.

    Returns:
        patch: The patch, copied, with what it measured and where it sits.
    """
    axes = observation.axes
    cell = tuple(int(at) for at in np.unravel_index(index, counts))
    origin = tuple(at * length for at, length in zip(cell, lengths, strict=True))
    window = tuple(
        slice(start, start + length)
        for start, length in zip(origin, lengths, strict=True)
    )
    taken = tuple(
        one for one, holds in zip(window, axes, strict=True) if holds == Axis.GROUND
    )
    spot = tuple(
        at for at, holds in zip(cell, axes, strict=True) if holds == Axis.GROUND
    )
    north_m, east_m = (float(held[spot]) for held in centres)
    ground_shape = tuple(
        length if holds == Axis.GROUND else 1
        for length, holds in zip(lengths, axes, strict=True)
    )
    measured = observation.measured[taken].reshape(ground_shape).copy()
    if Axis.DELAY in axes:
        # A sounder is placed by the rows it sounded, which is the patch's own cut.
        at = axes.index(Axis.DELAY)
        delay = observation.first_row + origin[at] + lengths[at] / 2
    else:
        # Every ground patch sits at the areoid row, so delay only places a sounder
        delay = float(sharad.AREOID_ROW)
    clutter = observation.clutter
    # What the instrument is conditioned on, cut as its values and flattened
    condition = np.array(
        [] if clutter is None else clutter[window] * measured, np.float32
    ).ravel()
    return Patch(
        values=observation.values[window].copy(),
        measured=measured,
        condition=condition,
        acquisition=acquisition,
        north_m=north_m,
        east_m=east_m,
        delay=delay,
        cell=cell,
    )


def patch_lengths(
    shape: Sequence[int],
    axes: Sequence[str],
    patchsize: Mapping[str, int],
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


def patch_shapes(
    by_tile: Mapping[str, Mapping[str, Sequence[ObservationMetadata]]],
    sizes: Mapping[str, Mapping[str, int]],
) -> dict[str, tuple[int, ...]]:
    """Return the shape of one patch of each instrument as the model reads it.

    Args:
        by_tile: The index rows of each instrument of each tile.
        sizes: How far a patch of each instrument runs along each axis it is cut on.

    Returns:
        shapes: The shape of one patch of each instrument, keyed as ODE names it.
    """
    shapes = {}
    for name, size in sizes.items():
        row = next(rows[name][0] for rows in by_tile.values() if name in rows)
        shapes[name] = patch_lengths(row.shape, row.axes, size)
    return shapes


def patch_arrays(
    patches: Sequence[Patch],
    name: str,
    shape: Sequence[int],
    axes: Sequence[str],
    acquisition: Sequence[str],
) -> dict[str, np.ndarray]:
    """Return one instrument's read patches as the arrays a model is handed.

    Args:
        patches: The patches, all of one instrument.
        name: The instrument, keyed as ODE names it.
        shape: The shape of one patch of it as the model reads it.
        axes: What each axis of its values holds.
        acquisition: The acquisition angles each patch carries, as the index names
            them.

    Returns:
        arrays: The patches under "values", "measured", "condition", "acquisition"
            and "position".
    """
    measured_shape = tuple(
        held if holds == Axis.GROUND else 1
        for held, holds in zip(shape, axes, strict=True)
    )
    width = math.prod(shape) if name == sharad.LAYOUT.instrument else 0
    return {
        "values": np.array([one.values for one in patches], np.float32).reshape(
            -1, *shape
        ),
        "measured": np.array([one.measured for one in patches], bool).reshape(
            -1, *measured_shape
        ),
        "condition": np.array([one.condition for one in patches], np.float32).reshape(
            len(patches), width
        ),
        "acquisition": np.array(
            [one.acquisition for one in patches], np.float32
        ).reshape(len(patches), len(acquisition)),
        "position": np.array(
            [[one.east_m, one.north_m, one.delay] for one in patches], np.float32
        ).reshape(-1, 3),
    }
