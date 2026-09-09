"""Drawing patches out of one crop, and keeping the ones that were measured."""

from __future__ import annotations

import math
import random

import numpy as np
from building.metadata.observation import ObservationMetadata

from dataset.config import tiles_of
from dataset.models.crop import Crop
from dataset.models.patch import Patch
from dataset.patches import grid, place

WAVELENGTHS = "wavelengths"

CANDIDATES = 4


def cut_patches(crop: Crop, record: ObservationMetadata, config: dict) -> list[Patch]:
    """Return the patches drawn from one crop that carry enough measurement.

    Args:
        crop: The crop to cut, whose axes say which of them are tiled.
        record: What the index says the crop is, which carries the ground it
            spans and when it was taken.
        config: The choices a read is made with, which say how far a patch of
            this instrument runs along each kind of axis, how many are drawn,
            and how much of one must be measured.

    Returns:
        patches: The patches drawn and kept, as many as the config asks for and
            fewer where too little of the crop was measured to fill it. A crop
            holds far more patches than a pass over it ever wants: one CTX scan
            runs to millions, so they are drawn rather than walked.
    """
    tiles = tiles_of(config, crop.instrument)
    lengths = grid.patch_lengths(crop.values.shape, crop.axes, tiles)
    counts = grid.patch_counts(crop.values.shape, crop.axes, tiles)
    held = math.prod(counts)
    if not held:
        return []
    asked = config["per_observation"]
    drawing = random.Random(f"{config['seed']}/{'/'.join(record.identity)}")
    patches = []
    for one in drawing.sample(range(held), min(held, asked * CANDIDATES)):
        if len(patches) == asked:
            break
        origin = tuple(
            int(at) * length
            for at, length in zip(np.unravel_index(one, counts), lengths, strict=True)
        )
        window = tuple(
            slice(start, start + length)
            for start, length in zip(origin, lengths, strict=True)
        )
        valid = crop.measured[tuple(window[at] for at in crop.ground)]
        if not valid.size or valid.mean() < config["keep_valid"]:
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
                wavelengths_nm=_wavelengths(crop, window),
                lon=lon,
                lat=lat,
                t_start=record.t_start,
                t_end=record.t_end,
            )
        )
    return patches


def _wavelengths(crop: Crop, window: tuple[slice, ...]) -> np.ndarray | None:
    """Return the centre wavelength of each band of one patch.

    Args:
        crop: The crop it was cut from, which stores them beside its values for
            an instrument whose wavelengths vary along the ground.
        window: What the patch keeps of each axis of the values.

    Returns:
        wavelengths: One wavelength per band, over the ground the patch keeps,
            and None for an instrument that stores none.
    """
    if WAVELENGTHS not in crop.beside:
        return None
    taken = dict(zip(crop.dims[crop.measurement], window, strict=True))
    held = crop.beside[WAVELENGTHS][
        tuple(taken.get(one, slice(None)) for one in crop.dims[WAVELENGTHS])
    ]
    return np.nanmean(held, axis=0)
