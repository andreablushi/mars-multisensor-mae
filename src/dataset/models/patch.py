"""One patch of one observation, and everything an embedding places it by."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Patch:
    """One patch cut from one observation, and what says where and what it is.

    Attributes:
        values: The patch's own values, in the observation's axis order.
        valid: Whether each sample is a measurement, broadcasting over the values.
        axes: What each axis of the values holds, in that same order.
        north_m: How far north of the tile centre the patch centre sits, in metres.
        east_m: How far east of it, in metres.
        delay: Which row of the radargram window the patch centre sounds at, its
            own cut for a sounder and the surface echo beneath it for the rest.
        north_span_m: How far the patch reaches northward, in metres.
        east_span_m: How far it reaches eastward, in metres.
        delay_span: How many rows it reaches across, which is none for a patch
            lying on the ground.
    """

    values: np.ndarray
    valid: np.ndarray
    axes: tuple[str, ...]
    north_m: float
    east_m: float
    delay: float
    north_span_m: float
    east_span_m: float
    delay_span: float
