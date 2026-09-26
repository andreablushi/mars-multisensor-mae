"""One stored observation, read back as the arrays and the description it holds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from building.common.layout import Axis
from building.preprocessing.common import relative_positioning
from building.preprocessing.common.models.position import Position
from common.models.tile import Tile


@dataclass(frozen=True, slots=True)
class Observation:
    """One cropped observation, its arrays and what says how to read them.

    Attributes:
        instrument: The instrument that took it, as ODE names it.
        identifier: What that instrument was asked for.
        measurement: What the values are called, as the archive publishes them.
        values: The values themselves, in the shape the instrument publishes.
        axes: What each axis of the values holds, in the array's own order.
        dims: What each axis of every array is called, keyed as it is written.
        measured: Whether each ground sample measures the tile, over ground alone.
        north: How far each sample sits north of the tile centre, as written.
        east: How far it sits eastward, holding the same.
        beside: What else the instrument stores, of what the read asked for.
        described: What the build wrote beside the arrays, which places them.
    """

    instrument: str
    identifier: str
    measurement: str
    values: np.ndarray
    axes: tuple[str, ...]
    dims: dict[str, tuple[str, ...]]
    measured: np.ndarray
    north: np.ndarray
    east: np.ndarray
    beside: dict[str, np.ndarray]
    described: dict

    @property
    def ground_axes(self) -> tuple[int, ...]:
        """Return which axes of the values are placed on the ground.

        Returns:
            ground_axes: Their positions in the array's own order.
        """
        return tuple(at for at, holds in enumerate(self.axes) if holds == Axis.GROUND)

    def distance_centre_m(
        self, taken: tuple[slice, ...] = ()
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return where the ground samples one cut keeps sit, in metres.

        Args:
            taken: What the cut keeps of each ground axis, empty for every sample.

        Returns:
            north: The ground metres north of the centre, one per sample kept.
            east: The ground metres east of it, in the same frame.
        """
        described = self.described
        separable = described["separable"]
        north, east = self.north, self.east
        if taken:
            north, east = (
                (north[taken[0]], east[taken[1]])
                if separable
                else (north[taken], east[taken])
            )
        grid = described["polar"]
        return relative_positioning.distance_centre_m(
            Position(north, east, separable, None if grid is None else tuple(grid)),
            Tile(described["band"], described["column"], **described["box"]),
        )
