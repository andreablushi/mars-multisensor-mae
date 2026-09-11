"""One stored observation, read back as the arrays and the description it holds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from building.common.layout import GROUND
from building.preprocessing.common import read
from building.preprocessing.common.models.relative_position import RelativePosition
from building.preprocessing.common.store import EAST, NORTH


@dataclass(frozen=True, slots=True)
class Observation:
    """One cropped observation, its arrays and what says how to read them.

    Attributes:
        instrument: The instrument that took it, as ODE names it.
        identifier: What that instrument was asked for.
        measurement: What the values are called, which is what the archive
            publishes them as and how the dims name their axes.
        values: The values themselves, in the shape the instrument publishes.
        axes: What each axis of the values holds, in the array's own order.
        dims: What each axis of every array it holds is called, keyed as
            that array is written.
        measured: Whether each ground sample is a measurement of the feature
            rather than a fill or ground outside its box, over the ground axes
            alone. A CTX scan spreads over a hundred million samples, so this is
            worked out once for every patch cut from it.
        north: How far each sample sits from the feature centre northward, as
            the build wrote it: degrees, or the metres of the grid it was
            placed on, and one value per line alone where that grid is
            separable.
        east: How far it sits eastward, holding the same.
        beside: What else the instrument stores, keyed as it is written.
        described: What the build wrote beside the arrays, which is what places
            them back on the ground.
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
    def ground(self) -> tuple[int, ...]:
        """Return which axes of the values are placed on the ground.

        Returns:
            ground: Their positions in the array's own order.
        """
        return tuple(at for at, holds in enumerate(self.axes) if holds == GROUND)

    def ground_metres(
        self, taken: tuple[slice, ...] = ()
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return where the ground samples one cut keeps sit, in metres.

        Args:
            taken: What the cut keeps of each ground axis, in the order those
                axes run, and empty for every sample of the observation.

        Returns:
            north: The ground metres north of the feature centre, one per
                sample the cut keeps.
            east: The ground metres east of it, in the same frame.
        """
        grid = self.described["polar"]
        held = RelativePosition(
            self.north,
            self.east,
            self.described["separable"],
            None if grid is None else tuple(grid),
        ).offsets(taken)
        return read.ground_metres({NORTH: held[0], EAST: held[1]}, self.described)
