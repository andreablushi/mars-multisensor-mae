"""The bands a model reads of each instrument, those every survey mode measures."""

from __future__ import annotations

import dataclasses

import numpy as np
from building.common.layout import Axis
from building.configs.crism import BANDS_NM, LAYOUT
from building.metadata.observation import ObservationMetadata

from dataset.models.observation import Observation

# fmt: off
MULTISPECTRAL_BANDS_NM = (
    408.715, 441.142, 532.000, 596.955, 648.954, 681.468, 707.489, 740.025,
    772.572, 798.619, 831.188, 857.252, 889.842, 922.445, 948.535, 981.159,
    1020.323, 1023.588, 1049.797, 1052.972, 1082.565, 1154.685, 1213.722, 1253.094,
    1259.658, 1266.221, 1279.350, 1331.876, 1371.284, 1397.563, 1430.419, 1469.857,
    1502.732, 1509.308, 1561.926, 1627.730, 1660.644, 1693.566, 1752.847, 1812.155,
    1878.084, 1930.851, 2122.309, 2142.131, 2168.565, 2208.225, 2234.672, 2254.511,
    2294.197, 2320.662, 2333.896, 2353.750, 2393.466, 2433.194, 2459.686, 2532.423,
    2605.032, 2631.447,
)
# fmt: on

READ_BANDS = {
    LAYOUT.instrument: tuple(BANDS_NM.index(band) for band in MULTISPECTRAL_BANDS_NM)
}


def measures_read_bands(name: str, record: ObservationMetadata) -> bool:
    """Return whether an observation measured every band read of its instrument.

    Args:
        name: The instrument, as ODE names it.
        record: The observation's index row.

    Returns:
        measured: True where every band read was measured, or no band is chosen.
    """
    kept = READ_BANDS.get(name, ())
    return all(record.band_valid_count[at] for at in kept)


def observation_read_bands(name: str, observation: Observation) -> Observation:
    """Return an observation holding only the bands read of its instrument.

    Args:
        name: The instrument, as ODE names it.
        observation: The observation, read whole.

    Returns:
        observation: The same observation, cut to the bands read where any are chosen.
    """
    if name not in READ_BANDS:
        return observation
    return dataclasses.replace(
        observation,
        values=np.take(
            observation.values,
            READ_BANDS[name],
            axis=observation.axes.index(Axis.WAVELENGTH),
        ),
    )


def read_band_patchsize(name: str) -> dict[str, int]:
    """Return how far a patch runs along the bands read of one instrument.

    Args:
        name: The instrument, as ODE names it.

    Returns:
        patchsize: The count of bands read along the wavelength axis, or empty.
    """
    return {Axis.WAVELENGTH: len(READ_BANDS[name])} if name in READ_BANDS else {}
