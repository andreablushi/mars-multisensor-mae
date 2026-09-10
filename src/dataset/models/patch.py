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
        valid: Whether each of its ground samples carries a measurement.
        axes: What each axis of the values holds, in that same order.
        origin: Where the patch starts along each axis of the observation.
        ground_sample_m: How much ground one sample spans along each ground
            axis, in the order those axes run.
        wavelengths_nm: The centre wavelength of each band in nanometres, and
            None for an instrument that holds no wavelength axis.
        north_m: How far north of the feature centre the patch centre sits,
            in metres.
        east_m: How far east of it, in metres.
        t_start: When the observation started, or None where none is published.
        t_end: When it ended, or None for the same reason.
    """

    instrument: str
    identifier: str
    values: np.ndarray
    valid: np.ndarray
    axes: tuple[str, ...]
    origin: tuple[int, ...]
    ground_sample_m: tuple[float, ...]
    wavelengths_nm: np.ndarray | None
    north_m: float
    east_m: float
    t_start: datetime | None
    t_end: datetime | None

    @property
    def measured_share(self) -> float:
        """Return how much of the patch carries a measurement.

        Returns:
            share: The measured fraction of its ground samples, 0 to 1.
        """
        return float(self.valid.mean()) if self.valid.size else 0.0
