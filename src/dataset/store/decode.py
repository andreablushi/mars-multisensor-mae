"""One crop's bytes, read back into the arrays and the description beside them."""

from __future__ import annotations

import io
import json

import numpy as np
from building.common.layout import GROUND
from building.preprocessing.common.store import EAST, INSIDE, META, NORTH, VALID, native

from dataset.models.crop import Crop


def read_crop(data: bytes) -> Crop:
    """Return one stored observation, read out of the bytes it was written as.

    Args:
        data: What the crop's own object holds.

    Returns:
        crop: The observation, its values in the machine's byte order and every
            sample counted as measured where the build stored no mask.
    """
    with np.load(io.BytesIO(data)) as held:
        described = json.loads(str(held[META]))
        arrays = {name: native(held[name]) for name in held.files if name != META}
    axes = tuple(described["axes"])
    values = arrays.pop(described["measurement"])
    ground = tuple(
        size for size, holds in zip(values.shape, axes, strict=True) if holds == GROUND
    )
    measured = np.ones(ground, dtype=bool)
    for name in (VALID, INSIDE):
        mask = arrays.pop(name, None)
        if mask is not None:
            measured &= mask
    polar = described["polar"]
    return Crop(
        instrument=described["instrument"],
        identifier=described["identifier"],
        measurement=described["measurement"],
        values=values,
        axes=axes,
        dims={name: tuple(held) for name, held in described["dims"].items()},
        measured=measured,
        north=arrays.pop(NORTH),
        east=arrays.pop(EAST),
        position_units=described["position_units"],
        polar=None if polar is None else tuple(polar),
        centre_lon=described["centre_lon"],
        centre_lat=described["centre_lat"],
        beside=arrays,
    )
