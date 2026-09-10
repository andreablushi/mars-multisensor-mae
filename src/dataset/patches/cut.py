"""Cutting one crop into the patches it holds that carry enough measurement."""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
from building.metadata.observation import ObservationMetadata

from dataset.models.crop import Crop
from dataset.models.patch import Patch
from dataset.patches import grid, place

WAVELENGTHS = "wavelengths"


def cut_patches(
    crop: Crop, record: ObservationMetadata, patchsize: int
) -> Iterator[Patch]:
    """Yield every patch of one crop.

    Args:
        crop: The crop to cut, whose axes say which of them are tiled.
        record: What the index says the crop is, which carries the ground it
            spans and when it was taken.
        patchsize: How far a patch of this instrument runs along every axis
            it is cut on.

    Yields:
        patch: Every whole patch of the crop, in the order its axes run, each
            carrying which of its samples were measured. One CTX scan holds
            tens of thousands, so they are yielded one at a time and never
            gathered: a feature runs to over a million.
    """
    lengths = grid.patch_lengths(crop.values.shape, crop.axes, patchsize)
    counts = grid.patch_counts(crop.values.shape, crop.axes, patchsize)
    for one in range(math.prod(counts)):
        origin = tuple(
            int(at) * length
            for at, length in zip(np.unravel_index(one, counts), lengths, strict=True)
        )
        window = tuple(
            slice(start, start + length)
            for start, length in zip(origin, lengths, strict=True)
        )
        valid = crop.measured[tuple(window[at] for at in crop.ground)]
        lon, lat = place.placement_of(crop, window)
        yield Patch(
            instrument=crop.instrument,
            identifier=crop.identifier,
            # Copied, so one patch does not hold the whole crop alive behind it.
            values=crop.values[window].copy(),
            valid=valid.copy(),
            axes=crop.axes,
            origin=origin,
            ground_sample_m=record.ground_sample_m,
            wavelengths_nm=_wavelengths(crop, window),
            lon=lon,
            lat=lat,
            t_start=record.t_start,
            t_end=record.t_end,
        )


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
