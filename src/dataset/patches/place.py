"""Where one patch of a crop sits on Mars, worked back from the offsets it carries."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from shared.maths import geodesy

from dataset.models.crop import Crop
from dataset.store.decode import EAST, NORTH

DEGREES = "degrees"


def placement_of(crop: Crop, window: Sequence[slice]) -> tuple[float, float]:
    """Return where the centre of one patch of a crop sits.

    Args:
        crop: The crop it was cut from, which carries the offset every sample of
            it stands at and the centre they stand from.
        window: What the patch keeps of each axis of the values, in the axes'
            own order.

    Returns:
        lon: The longitude of the patch centre in degrees, -180 to 180.
        lat: Its latitude in degrees.
    """
    held = dict(zip(crop.dims[crop.measurement], window, strict=True))

    def middle(name: str) -> float:
        """Return one offset array where the middle of the patch falls on it."""
        taken = tuple(
            (held[one].start + held[one].stop) // 2 for one in crop.dims[name]
        )
        return float(getattr(crop, name)[taken])

    down, across = middle(NORTH), middle(EAST)
    if crop.position_units == DEGREES:
        return (
            float(geodesy.normalise_longitude(crop.centre_lon + across)),
            crop.centre_lat + down,
        )
    centre_x, centre_y = geodesy.stereographic_forward(
        crop.centre_lon, crop.centre_lat, *crop.polar
    )
    lon, lat = geodesy.stereographic_inverse(
        np.array(across + centre_x), np.array(down + centre_y), *crop.polar
    )
    return float(lon), float(lat)
