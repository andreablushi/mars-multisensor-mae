"""Pairing every patch with the height of the ground it stands on."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from building.common.layout import ELEVATION
from building.metadata.observation import ObservationMetadata

from dataset.models.patch import Patch
from dataset.store import DatasetBuild

HEIGHTS = "elevation"


@dataclass(frozen=True, slots=True)
class Elevation:
    """Every height the elevation instrument measured over one feature, placed.

    Attributes:
        north: How far north of the feature centre each measured sample sits,
            in metres. (N,)
        east: How far east of it, in the same metres. (N,)
        height: How high above the areoid the ground stands there, in metres. (N,)
    """

    north: np.ndarray
    east: np.ndarray
    height: np.ndarray

    def at(self, north_m: float, east_m: float) -> float:
        """Return the height of the measured sample nearest one point.

        Args:
            north_m: How far north of the feature centre the point sits.
            east_m: How far east of it.

        Returns:
            height_m: How high above the areoid the nearest sample stands.
        """
        nearest = np.argmin((self.north - north_m) ** 2 + (self.east - east_m) ** 2)
        return float(self.height[nearest])


def read_elevation(
    observations: Sequence[ObservationMetadata], build: DatasetBuild
) -> Elevation:
    """Return every height measured over one feature, pooled over its tiles.

    Args:
        observations: The feature's index rows of the elevation instrument.
        build: The published build the observations are read from.

    Returns:
        elevation: Every measured sample of every one of them, placed.

    Raises:
        ValueError: When none of them measured anything.
    """
    north, east, height = [], [], []
    for record in observations:
        observation = build.read_observation(record.path)
        measured = observation.measured
        north.append(observation.north[measured])
        east.append(observation.east[measured])
        height.append(observation.values[measured].astype(np.float64))
    if not sum(one.size for one in height):
        raise ValueError(
            f"nothing measured over {[one.identity for one in observations]}"
        )
    return Elevation(
        np.concatenate(north), np.concatenate(east), np.concatenate(height)
    )


def patch_elevation(patch: Patch, elevation: Elevation) -> float:
    """Return how high the centre of one patch stands.

    Args:
        patch: The patch, which carries its own heights where its instrument
            sounds down through the ground.
        elevation: Where the ground stands over the feature, for every other
            instrument.

    Returns:
        height_m: Metres above the areoid, the middle of the heights a sounding
            patch spans and the nearest measured ground under any other.
    """
    if ELEVATION in patch.axes:
        return float(np.mean(patch.beside[HEIGHTS]))
    return elevation.at(patch.north_m, patch.east_m)
