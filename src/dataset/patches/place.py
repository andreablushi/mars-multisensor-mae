"""Where one patch of an observation sits on Mars, worked back from its offsets."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from building.preprocessing.common.store import DEGREES, EAST, NORTH
from shared.maths import geodesy

from dataset.models.observation import Observation


def placement_of(
    observation: Observation, window: Sequence[slice]
) -> tuple[float, float]:
    """Return where the centre of one patch of an observation sits.

    Args:
        observation: The observation it was cut from, which carries the offset
            every sample of it stands at and the centre they stand from.
        window: What the patch keeps of each axis of the values, in the axes'
            own order.

    Returns:
        lon: The longitude of the patch centre in degrees, -180 to 180.
        lat: Its latitude in degrees.
    """
    held = dict(zip(observation.dims[observation.measurement], window, strict=True))

    def middle(name: str) -> float:
        """Return one offset array where the middle of the patch falls on it."""
        taken = tuple(
            (held[one].start + held[one].stop) // 2 for one in observation.dims[name]
        )
        return float(getattr(observation, name)[taken])

    down, across = middle(NORTH), middle(EAST)
    if observation.position_units == DEGREES:
        return (
            float(geodesy.normalise_longitude(observation.centre_lon + across)),
            observation.centre_lat + down,
        )
    centre_x, centre_y = geodesy.stereographic_forward(
        observation.centre_lon, observation.centre_lat, *observation.polar
    )
    lon, lat = geodesy.stereographic_inverse(
        np.array(across + centre_x), np.array(down + centre_y), *observation.polar
    )
    return float(lon), float(lat)
