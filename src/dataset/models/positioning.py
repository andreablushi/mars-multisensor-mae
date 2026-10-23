"""Where the samples of one observation sit around their tile centre."""

from __future__ import annotations

import numpy as np
from building.preprocessing.common import geometry, relative_positioning
from building.preprocessing.common.models.position import Position
from building.preprocessing.common.store import STORED
from common.maths.geodesy import SPHEROID, PolarGrid
from common.models.tile import Tile

from dataset.models.observation import Observation


# It's done there for memory reasons
def metres_from_centre(
    observation: Observation, taken: tuple[slice, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Return how far north and east of its tile centre every sample one cut keeps sits.

    Args:
        observation: The stored observation the cut is taken of.
        taken: What the cut keeps of each ground axis.

    Returns:
        north: The ground metres north of the centre, one per sample kept.
        east: The ground metres east of it, in the same frame.
    """
    described = observation.described
    separable = described["separable"]
    north, east = observation.north, observation.east
    if separable:
        north, east = north[taken[0]], east[taken[1]]
    else:
        north, east = north[taken], east[taken]
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
