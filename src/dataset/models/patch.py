"""One patch of one observation, and everything an embedding places it by."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass(frozen=True, slots=True)
class Patch:
    """One patch cut from one observation, and what says where and what it is.

    Attributes:
        instrument: The instrument that took it, as ODE names it.
        identifier: The observation it was cut from.
        values: The patch's own values, in the observation's axis order.
        valid: Whether each sample is a measurement, broadcasting over the values.
        axes: What each axis of the values holds, in that same order.
        origin: Where the patch starts along each axis of the observation.
        north_m: How far north of the tile centre the patch centre sits, in metres.
        east_m: How far east of it, in metres.
        height_m: How high above the areoid the patch centre stands, in metres,
            read off the delay for a sounder and off the elevation for the rest.
        north_span_m: How far the patch reaches northward, in metres.
        east_span_m: How far it reaches eastward, in metres.
        height_span_m: How far it reaches in height, in metres, which is nothing
            for a patch lying on the ground.
        t_start: When the observation started, or None where none is published.
        t_end: When it ended, or None for the same reason.
    """

    instrument: str
    identifier: str
    values: np.ndarray
    valid: np.ndarray
    axes: tuple[str, ...]
    origin: tuple[int, ...]
    north_m: float
    east_m: float
    height_m: float
    north_span_m: float
    east_span_m: float
    height_span_m: float
    t_start: datetime | None
    t_end: datetime | None
