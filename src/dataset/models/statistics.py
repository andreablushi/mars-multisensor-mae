"""What one instrument's values run to, pooled over every crop the index names."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Statistics:
    """What the values of one instrument run to, over the whole of a build.

    Attributes:
        instrument: The instrument they were pooled over, as ODE names it.
        count: How many measured values they were pooled from.
        mean: Their mean.
        deviation: Their standard deviation.
        band_mean: The mean of each band, and None for an instrument that holds
            no wavelength axis or a build that published none.
        band_deviation: The standard deviation of each band, or None for the
            same reasons.
    """

    instrument: str
    count: int
    mean: float
    deviation: float
    band_mean: np.ndarray | None = None
    band_deviation: np.ndarray | None = None
