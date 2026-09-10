"""One stored observation, read back as the arrays and the description it holds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from building.common.layout import GROUND


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
        north: How far each sample sits from the feature centre, northward.
        east: How far it sits eastward.
        position_units: Whether those two are degrees or a projection's metres.
        polar: The projection they were placed on, and None where they are
            degrees.
        centre_lon: The longitude they stand from, in degrees.
        centre_lat: The latitude they stand from, in degrees.
        beside: What else the instrument stores, keyed as it is written.
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
    position_units: str
    polar: tuple[float, bool, float] | None
    centre_lon: float
    centre_lat: float
    beside: dict[str, np.ndarray]

    @property
    def ground(self) -> tuple[int, ...]:
        """Return which axes of the values are placed on the ground.

        Returns:
            ground: Their positions in the array's own order.
        """
        return tuple(at for at, holds in enumerate(self.axes) if holds == GROUND)
