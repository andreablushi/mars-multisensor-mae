"""What each band of a spectral instrument is centred on, beside the patches it cuts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from building.common.layout import WAVELENGTH
from building.configs import crism

# The centre wavelength of every band of each spectral instrument, in nanometres.
# A build reads every observation onto its instrument's own grid, so a band holds
# the same wavelength whichever observation a patch was cut from.
NANOMETRES = {crism.LAYOUT.instrument: crism.WAVELENGTHS_NM}


def band_wavelengths(
    shapes: Mapping[str, tuple[int, ...]], axes: Mapping[str, Sequence[str]]
) -> dict[str, tuple[float, ...]]:
    """Return what each band of every spectral instrument read is centred on.

    Args:
        shapes: The shape of one patch of each instrument, keyed as ODE names
            it.
        axes: What each axis of that instrument's values holds, in that same
            order.

    Returns:
        nanometres: The centre wavelength of every band, in the order the
            wavelength axis of a patch runs, keyed as ODE names the instrument.
            An instrument whose patches hold no wavelength is not in it.

    Raises:
        ValueError: When an instrument holds a wavelength axis the table does
            not cover, which is a build whose bands are no longer the ones
            written here.
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
