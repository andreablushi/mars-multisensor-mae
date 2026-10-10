"""One patch of one observation, and everything an embedding places it by."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Patch:
    """One patch cut from one observation, and what says where and what it is.

    Attributes:
        values: The patch's own values, in the observation's axis order.
        measured: Whether each sample is a measurement, broadcasting over the values.
        axes: What each axis of the values holds, in that same order.
        north_m: How far north of the tile centre the patch centre sits, in metres.
        east_m: How far east of it, in metres.
        delay: Which row of the radargram window the patch centre sounds at, its
            own cut for a sounder and the surface echo beneath it for the rest.
        cell: Which patch of its observation it is, counted along each axis.
    """

    values: np.ndarray
    measured: np.ndarray
    axes: tuple[str, ...]
    north_m: float
    east_m: float
    delay: float
    cell: tuple[int, ...]
