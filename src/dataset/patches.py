"""Cutting one observation into the patches a transformer reads it as."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from building.common.layout import Axis
from building.configs import sharad
from building.metadata.observation import ObservationMetadata

from dataset.models.observation import Observation
from dataset.models.patch import Patch
from dataset.models.positioning import metres_from_centre

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
            observation = build.read_observation(record.path, build.conditions(name))
            angles = np.array(
                [getattr(record.acquisition, one) for one in build.acquired(name)],
                np.float32,
            )
            shape = observation.values.shape
            lengths = patch_lengths(shape, record.axes, size)
            counts = tuple(
                length // patch for length, patch in zip(shape, lengths, strict=True)
            )
            for at in range(math.prod(counts)):
                patch = cut_patch(observation, at, lengths, counts, angles)
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
) -> Patch:
    """Return one whole patch of one observation.

    Args:
        observation: The observation, read whole.
        index: Which patch, counting whole ones as the axes run, the last fastest.
        lengths: How far one patch of it runs along each axis.
        counts: How many whole patches each of those axes holds.
        acquisition: The observation's acquisition angles the model reads.

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
    measured = observation.measured[taken].reshape(ground_shape).copy()
    if Axis.DELAY in axes:
        # A sounder is placed by the rows it sounded, which is the patch's own cut.
        at = axes.index(Axis.DELAY)
        delay = origin[at] + lengths[at] / 2
    else:
        # Every ground patch sits at the areoid row, so delay only places a sounder
        delay = float(sharad.AREOID_ROW)
    # What the instrument is conditioned on, cut as its values and flattened
    condition = np.array(
        [observation.beside[one][window] * measured for one in observation.beside],
        np.float32,
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


def patch_arrays(
    patches: Sequence[Patch],
    shape: Sequence[int],
    axes: Sequence[str],
    width: int,
    acquired: int,
) -> dict[str, np.ndarray]:
    """Return one instrument's read patches as the arrays a model is handed.

    Args:
        patches: The patches, all of one instrument.
        shape: The shape of one patch of it as the model reads it.
        axes: What each axis of its values holds.
        width: How many values each patch is conditioned on.
        acquired: How many acquisition angles each patch carries.

    Returns:
        arrays: The patches under "values", "measured", "condition", "acquisition"
            and "position".
    """
    measured_shape = tuple(
        held if holds == Axis.GROUND else 1
        for held, holds in zip(shape, axes, strict=True)
    )
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
        ).reshape(len(patches), acquired),
        "position": np.array(
            [[one.east_m, one.north_m, one.delay] for one in patches], np.float32
        ).reshape(-1, 3),
    }


def cropped_patch_arrays(
    sample: dict[str, dict[str, np.ndarray]],
    budget: Mapping[str, int],
    draw: random.Random,
) -> dict[str, dict[str, np.ndarray]]:
    """Return a tile's patch arrays, each instrument cut to the ones nearest a centre.

    Args:
        sample: Each instrument's patch arrays over the tile.
        budget: How many patches of each instrument are kept.
        draw: What picks the centre.

    Returns:
        sample: Each instrument's kept patch arrays, nearest the centre first.
    """
    # Each ground spot once, whether a sounder's delay is cut into patches or not
    grounds = np.unique(
        np.concatenate([arrays["position"][:, :2] for arrays in sample.values()]),
        axis=0,
    )
    # One patch spot of any instrument is the centre every instrument is cut around
    centre = grounds[draw.randrange(len(grounds))]
    cropped = {}
    for name, arrays in sample.items():
        apart = np.linalg.norm(arrays["position"][:, :2] - centre, axis=1)
        kept = np.argsort(apart, kind="stable")[: budget[name]]
        cropped[name] = {key: array[kept] for key, array in arrays.items()}
    return cropped
