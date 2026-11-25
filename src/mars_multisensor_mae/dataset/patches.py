"""Cutting one observation into the patches a transformer reads it as."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from mars_multisensor_dataset.building.common.layout import Axis
from mars_multisensor_dataset.building.configs import sharad
from mars_multisensor_dataset.building.metadata.observation import ObservationMetadata

from mars_multisensor_mae.dataset.models.observation import Window
from mars_multisensor_mae.dataset.models.patch import (
    InstrumentPatches,
    Patch,
    PatchLayout,
    TileSample,
)
from mars_multisensor_mae.dataset.store import DatasetBuild


def cut_tile_patches(
    rows: Mapping[str, Sequence[ObservationMetadata]],
    build: DatasetBuild,
    layout: PatchLayout,
    windows: Mapping[str, Window] | None = None,
) -> dict[str, list[Patch]]:
    """Return every whole patch of every observation one tile holds, by instrument.

    Args:
        rows: The tile's index rows of each instrument, keyed as ODE names it.
        build: The published build the observations are read from, a windowed
            instrument from its own.
        layout: How each instrument is cut into patches, a patch's shape its length
            along each axis.
        windows: The part of a windowed instrument's observations read, keyed by
            instrument, the rest read whole.

    Returns:
        read: The patches of each instrument measuring anything, in the order the
            axes run, the last fastest.
    """
    read: dict[str, list[Patch]] = {}
    for name, lengths in layout.shapes.items():
        held: list[Patch] = []
        for record in rows[name]:
            observation = build.windowed.get(name, build).read_observation(
                record.path, (windows or {}).get(name)
            )
            axes = observation.axes
            angles = np.array(
                [getattr(record.acquisition, one) for one in build.acquisition[name]],
                np.float32,
            )
            counts = tuple(
                length // patch
                for length, patch in zip(observation.values.shape, lengths, strict=True)
            )
            if not math.prod(counts):
                continue
            ground = [at for at, holds in enumerate(axes) if holds == Axis.GROUND]
            centres = observation.patch_metres_from_centre(
                [lengths[at] for at in ground], [counts[at] for at in ground]
            )
            ground_shape = tuple(
                length if holds == Axis.GROUND else 1
                for length, holds in zip(lengths, axes, strict=True)
            )
            for index in range(math.prod(counts)):
                cell = np.unravel_index(index, counts)
                window = tuple(
                    slice(at * length, (at + 1) * length)
                    for at, length in zip(cell, lengths, strict=True)
                )
                measured = observation.measured[
                    tuple(window[at] for at in ground)
                ].reshape(ground_shape)
                if not measured.any():
                    continue
                north_m, east_m = (
                    held_at[tuple(cell[at] for at in ground)] for held_at in centres
                )
                if Axis.DELAY in axes:
                    # A sounder is placed by the rows it sounded,
                    # which is the patch's own cut.
                    at = axes.index(Axis.DELAY)
                    delay = observation.first_row + window[at].start + lengths[at] / 2
                else:
                    # Every ground patch sits at the areoid row,
                    # so delay only places a sounder
                    delay = sharad.AREOID_ROW
                clutter = observation.clutter
                # What the instrument is conditioned on, cut as its values and flattened
                condition = (
                    np.empty(0, np.float32)
                    if clutter is None
                    else clutter[window].ravel()
                )
                held.append(
                    Patch(
                        values=observation.values[window],
                        measured=measured,
                        condition=condition,
                        acquisition=angles,
                        north_m=north_m,
                        east_m=east_m,
                        delay=delay,
                        cell=cell,
                    )
                )
        read[name] = held
    return read


def drawn_arrays(
    build: DatasetBuild,
    name: str,
    rows: Sequence[ObservationMetadata],
    layout: PatchLayout,
) -> dict[str, np.ndarray]:
    """Return some observations of one instrument as the arrays drawn.

    Args:
        build: The build they are read from.
        name: The instrument, as ODE names it.
        rows: The observations' index rows, in the order they are laid.
        layout: How each instrument is cut into patches, the instrument among them.

    Returns:
        arrays: Their patches as InstrumentPatches names them, plus "cell", the
            z-scores' "clip", "mean" and "std", "spacing", "observation" and
            "identifier".
    """
    alone = layout.only([name])
    read = [cut_tile_patches({name: [row]}, build, alone)[name] for row in rows]
    patches = [patch for held in read for patch in held]
    counts = [len(held) for held in read]
    stats = build.read_stats()[name]
    stacked = InstrumentPatches.stacked(
        patches, name, layout.shapes[name], rows[0].axes, build.acquisition[name]
    )
    normalised = build.normalised_sample(TileSample({name: stacked}))
    return normalised.instruments[name]._asdict() | {
        "cell": np.array([one.cell for one in patches]),
        "clip": np.array(build.clips[name], dtype=float),
        "mean": np.array(stats["mean"], dtype=float),
        "std": np.array(stats["std"], dtype=float),
        "spacing": np.repeat([row.sample_spacing_m for row in rows], counts, axis=0),
        "observation": np.repeat(np.arange(len(rows)), counts),
        "identifier": np.array([row.identifier for row in rows]),
    }
