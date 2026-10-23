"""One stored observation, read back as the arrays and the description it holds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from building.common.layout import Axis


@dataclass(frozen=True, slots=True)
class Observation:
    """One cropped observation, its arrays and what says how to read them.

    Attributes:
        values: The values themselves, in the shape the instrument publishes.
        axes: What each axis of the values holds, in the array's own order.
        measured: Whether each ground sample measures the tile, over ground alone.
        north: How far each sample sits north of the tile centre, as written.
        east: How far it sits eastward, holding the same.
        beside: What else the instrument stores, of what the read asked for.
        described: What the build wrote beside the arrays, which places them.
    """

    values: np.ndarray
    axes: tuple[str, ...]
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
