"""The bands a model reads of each instrument, those every survey mode measures."""

from __future__ import annotations

import dataclasses

import numpy as np
from building.common.layout import Axis
from building.configs.crism import BANDS_NM, LAYOUT, MULTISPECTRAL_BANDS_NM
from building.metadata.observation import ObservationMetadata

from dataset.models.observation import Observation

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
