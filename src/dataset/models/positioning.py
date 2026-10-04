"""Where the samples of one observation sit around their tile centre, and its delays."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from building.metadata.observation import ObservationMetadata
from building.preprocessing.common import geometry, relative_positioning
from building.preprocessing.common.models.position import Position
from building.preprocessing.common.store import STORED
from common.maths.geodesy import SPHEROID, PolarGrid
from common.models.tile import Tile

from dataset.models.observation import Observation

if TYPE_CHECKING:
    from dataset.store import DatasetBuild


# It's done there for memory reasons
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
