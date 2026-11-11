"""One stored observation, read back as the arrays and the description it holds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mars_multisensor_dataset.building.preprocessing.common import (
    geometry,
    relative_positioning,
)
from mars_multisensor_dataset.building.preprocessing.common.models.position import (
    Position,
)
from mars_multisensor_dataset.building.preprocessing.common.store import STORED
from mars_multisensor_dataset.common.maths.geodesy import SPHEROID, PolarGrid
from mars_multisensor_dataset.common.models.tile import Tile


@dataclass(frozen=True, slots=True)
class Observation:
    """One cropped observation, its arrays and what says how to read them.

    Attributes:
        values: The values themselves, in the shape the instrument publishes.
        axes: What each axis of the values holds, in the array's own order.
        measured: Whether each ground sample measures the tile, over ground alone.
        north: How far each sample sits north of the tile centre, as written.
        east: How far it sits eastward, holding the same.
        clutter: The cluttergram on the values' grid for SHARAD, else None.
        described: What the build wrote beside the arrays, which places them.
        first_row: Which archive row a sounder's first delay row is, 0 when uncropped.
    """

    values: np.ndarray
    axes: tuple[str, ...]
    measured: np.ndarray
    north: np.ndarray
    east: np.ndarray
    clutter: np.ndarray | None
    described: dict
    first_row: int

    def mean_metres_from_centre(self, taken: tuple[slice, ...]) -> tuple[float, float]:
        """Return how far north and east of the tile centre one cut sits on average.

        Args:
            taken: What the cut keeps of each ground axis.

        Returns:
            north_m: The mean ground metres north of the centre.
            east_m: The mean ground metres east of it.
        """
        described = self.described
        separable = described["separable"]
        if separable:
            # A separable grid is regular, so its corners place the patch as a whole
            corners = tuple(
                slice(one.start, one.stop, one.stop - one.start - 1 or 1)
                for one in taken
            )
            north, east = self.north[corners[0]], self.east[corners[1]]
        else:
            north, east = self.north[taken], self.east[taken]
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
        return float(np.mean(northing)), float(np.mean(easting))
