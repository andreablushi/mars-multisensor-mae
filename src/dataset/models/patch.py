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
        condition: What the model reads beside the values, flattened, zero where
            unmeasured: the cluttergram for SHARAD, else nothing.
        acquisition: The acquisition angles of its observation the model reads, in
            degrees, NaN where the index holds none.
        north_m: How far north of the tile centre the patch centre sits, in metres.
        east_m: How far east of it, in metres.
        delay: Which row of the radargram window the patch centre sounds at, its
            own cut for a sounder and the areoid row for the rest.
        cell: Where the patch sits on its observation's grid of patches, per axis.
    """

    values: np.ndarray
    measured: np.ndarray
    condition: np.ndarray
    acquisition: np.ndarray
    north_m: float
    east_m: float
    delay: float
    cell: tuple[int, ...]
