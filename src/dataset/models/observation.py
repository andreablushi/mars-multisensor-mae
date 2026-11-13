"""One stored observation, read back as the arrays and the description it holds."""

from __future__ import annotations

from collections.abc import Sequence
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

    def patch_metres_from_centre(
        self, lengths: Sequence[int], counts: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return how far north and east of the tile centre every whole patch sits.

        Args:
            lengths: How far one patch runs along each ground axis.
            counts: How many whole patches each ground axis holds.

        Returns:
            north_m: Each patch's mean ground metres north of the centre, one per
                patch along every ground axis.
            east_m: Its mean ground metres east of it, holding the same.
        """
        described = self.described
        separable = described["separable"]
        if separable:
            # A separable grid is regular, so its corners place each patch as a whole
            north, east = (
                held[
                    np.add.outer(
                        np.arange(count) * length, sorted({0, length - 1})
                    ).ravel()
                ]
                for held, length, count in zip(
                    (self.north, self.east), lengths, counts, strict=True
                )
            )
        else:
            whole = tuple(
                slice(length * count)
                for length, count in zip(lengths, counts, strict=True)
            )
            north, east = self.north[whole], self.east[whole]
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
        # Each patch's own samples side by side, in the order they are stored
        split = [
            size
            for count, held in zip(counts, position.sizes, strict=True)
            for size in (count, held // count)
        ]
        order = (*range(0, len(split), 2), *range(1, len(split), 2))
        return tuple(
            np.ascontiguousarray(held.reshape(split).transpose(order))
            .reshape(*counts, -1)
            .mean(-1)
            for held in (northing, easting)
        )
