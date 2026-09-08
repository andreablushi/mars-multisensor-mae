"""Drawing patches out of one crop, and keeping the ones that were measured."""

from __future__ import annotations

import random

import numpy as np
from building.common.layout import WAVELENGTH
from building.metadata.observation import ObservationMetadata

from dataset.models.crop import Crop
from dataset.models.patch import Patch
from dataset.models.settings import Settings
from dataset.patches import grid, place
from dataset.seed import seeded_number

WAVELENGTHS = "wavelengths"

CANDIDATES = 4


def cut_patches(
    crop: Crop, record: ObservationMetadata, settings: Settings, epoch: int = 0
) -> list[Patch]:
    """Return the patches drawn from one crop that carry enough measurement.

    Args:
        crop: The crop to cut, whose axes say which of them are tiled.
        record: What the index says the crop is, which carries the ground it
            spans and when it was taken.
        settings: The settled choices, which say how far a patch of this
            instrument runs along each kind of axis, how many are drawn, and
            how much of one must be measured.
        epoch: Which pass over the dataset this is, so one crop is cut
            differently from one pass to the next and the same within one.

    Returns:
        patches: The patches drawn and kept, as many as the config asks for and
            fewer where too little of the crop was measured to fill it. A crop
            holds far more patches than a pass over it ever wants: one CTX scan
            runs to millions, so they are drawn rather than walked.
    """
    shape = crop.values.shape
    tiles = settings.tiles_of(crop.instrument)
    lengths = grid.patch_lengths(shape, crop.axes, tiles)
    counts = grid.patch_counts(shape, crop.axes, tiles)
    held = grid.patch_count(shape, crop.axes, tiles)
    if not held:
        return []
    dims = crop.dims[crop.measurement]
    spectral = WAVELENGTH in crop.axes
    measured = crop.measured
    drawing = random.Random(
        seeded_number("/".join(record.identity), settings.seed + epoch)
    )
    asked = settings.per_observation
    drawn = drawing.sample(range(held), min(held, asked * CANDIDATES))

    def alongside(name: str, window: tuple[slice, ...]) -> np.ndarray:
        """Return what one array beside the values keeps of the same patch."""
        taken = dict(zip(dims, window, strict=True))
        return crop.beside[name][
            tuple(taken.get(one, slice(None)) for one in crop.dims[name])
        ]

    patches = []
    for one in drawn:
        if len(patches) == asked:
            break
        origin = grid.origin_of(one, counts, lengths)
        window = tuple(
            slice(start, start + length)
            for start, length in zip(origin, lengths, strict=True)
        )
        valid = measured[tuple(window[at] for at in crop.ground)]
        if not valid.size or valid.mean() < settings.keep_valid:
            continue
        lon, lat = place.placement_of(crop, window)
        patches.append(
            Patch(
                instrument=crop.instrument,
                identifier=crop.identifier,
                values=crop.values[window],
                valid=valid,
                axes=crop.axes,
                origin=origin,
                ground_sample_m=record.ground_sample_m,
                wavelengths_nm=(
                    np.nanmean(alongside(WAVELENGTHS, window), axis=0)
                    if spectral and WAVELENGTHS in crop.beside
                    else None
                ),
                lon=lon,
                lat=lat,
                t_start=record.t_start,
                t_end=record.t_end,
            )
        )
    return patches
