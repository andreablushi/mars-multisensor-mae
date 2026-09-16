"""What each band of a spectral instrument is centred on, beside the patches it cuts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from building.common.layout import WAVELENGTH
from building.configs import crism

NANOMETRES = {crism.LAYOUT.instrument: crism.WAVELENGTHS_NM}


def band_wavelengths(
    shapes: Mapping[str, tuple[int, ...]], axes: Mapping[str, Sequence[str]]
) -> dict[str, tuple[float, ...]]:
    """Return what each band of every spectral instrument read is centred on.

    Args:
        shapes: The shape of one patch of each instrument, keyed as ODE names it.
        axes: What each axis of that instrument's values holds, in that same order.

    Returns:
        nanometres: Every band's centre wavelength, keyed as ODE names the sensor.

    Raises:
        ValueError: When a sensor holds a wavelength axis the table does not cover.
    """
    read = {}
    for name, shape in shapes.items():
        held = [
            size
            for size, holds in zip(shape, axes[name], strict=True)
            if holds == WAVELENGTH
        ]
        if not held:
            continue
        (bands,) = held
        centres = NANOMETRES.get(name, ())
        if len(centres) != bands:
            raise ValueError(
                f"{name} cuts patches of {bands} bands, "
                f"and {len(centres)} wavelengths are written for it"
            )
        read[name] = centres
    return read
