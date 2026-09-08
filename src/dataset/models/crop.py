"""One stored observation, read back as the arrays and the description it holds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from building.common.layout import GROUND


@dataclass(frozen=True, slots=True)
class Crop:
    """One cropped observation, its arrays and what says how to read them.

    Attributes:
        instrument: The instrument that took it, as ODE names it.
        identifier: What that instrument was asked for.
        measurement: What the values are called, which is what the archive
            publishes them as.
        values: The values themselves, in the shape the instrument publishes.
        axes: What each axis of the values holds, in the array's own order.
        dims: What each axis of every array it holds is called, keyed as
            that array is written.
        valid: Whether each ground sample is a measurement rather than a fill,
            over the ground axes alone.
        inside: Whether each ground sample falls in the feature's box rather than
            only in the rectangle around it, over the same axes.
        north: How far each sample sits from the feature centre, northward.
        east: How far it sits eastward.
        separable: Whether those two hold one axis each rather than a value per
            sample.
        position_units: Whether they are degrees or the metres of a projection.
        polar: The projection they were placed on, and None where they are
            degrees.
        centre_lon: The longitude they stand from, in degrees.
        centre_lat: The latitude they stand from, in degrees.
        beside: What else the instrument stores, keyed as it is written.
        label: The merged archive label of the products it was cut from.
    """

    instrument: str
    identifier: str
    measurement: str
    values: np.ndarray
    axes: tuple[str, ...]
    dims: dict[str, tuple[str, ...]]
    valid: np.ndarray
    inside: np.ndarray
    north: np.ndarray
    east: np.ndarray
    separable: bool
    position_units: str
    polar: tuple[float, bool, float] | None
    centre_lon: float
    centre_lat: float
    beside: dict[str, np.ndarray]
    label: dict[str, str]

    @property
    def ground(self) -> tuple[int, ...]:
        """Return which axes of the values are placed on the ground.

        Returns:
            ground: Their positions in the array's own order.
        """
        return tuple(at for at, holds in enumerate(self.axes) if holds == GROUND)

    @property
    def measured(self) -> np.ndarray:
        """Return which ground samples carry a measurement of the feature.

        Returns:
            measured: True where the sample was measured and falls in the box.
                A CTX scan spreads over a hundred million samples, so a caller
                cutting many patches works this out once and not once each.
        """
        return self.valid & self.inside
