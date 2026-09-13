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
        valid: Whether each sample of the patch is a measurement, holding its
            ground axes and its wavelength axis and one along every other, so
            it broadcasts over the values. A band the observation never
            measured was filled rather than read, and carries none.
        axes: What each axis of the values holds, in that same order.
        origin: Where the patch starts along each axis of the observation.
        ground_sample_m: How much ground one sample spans along each ground
            axis, in the order those axes run.
        beside: What the instrument stores beside its values, keyed as it is
            written and cut to the patch. A SHARAD patch carries the elevation
            of every delay, and every other instrument nothing.
        north_m: How far north of the feature centre the patch centre sits,
            in metres.
        east_m: How far east of it, in metres.
        height_m: How high above the areoid the patch centre stands, in
            metres: the middle of the heights a sounding patch spans, and the
            nearest measured ground under any other.
        north_span_m: How far the patch reaches northward, in metres.
        east_span_m: How far it reaches eastward, in metres.
        height_span_m: How far it reaches in height, in metres, which a
            sounding patch spans and a surface patch does not.
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
    beside: dict[str, np.ndarray]
    north_m: float
    east_m: float
    height_m: float
    north_span_m: float
    east_span_m: float
    height_span_m: float
    t_start: datetime | None
    t_end: datetime | None

    @property
    def measured_share(self) -> float:
        """Return how much of the patch carries a measurement.

        Returns:
            share: The measured fraction of the samples the mask holds, 0 to 1.
        """
        return float(self.valid.mean()) if self.valid.size else 0.0
