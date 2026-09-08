"""One crop's bytes, read back into the arrays and the description beside them."""

from __future__ import annotations

import io
import json

import numpy as np
from building.common.layout import GROUND
from building.preprocessing.common.store import EAST, INSIDE, META, NORTH, VALID

from dataset.models.crop import Crop


def read_crop(data: bytes) -> Crop:
    """Return one stored observation, read out of the bytes it was written as.

    Args:
        data: What the crop's own object holds.

    Returns:
        crop: The observation, its values in the machine's byte order and its
            masks standing at every sample where the build stored none.
    """
    with np.load(io.BytesIO(data)) as held:
        described = json.loads(str(held[META]))
        arrays = {name: _native(held[name]) for name in held.files if name != META}
    axes = tuple(described["axes"])
    values = arrays.pop(described["measurement"])
    ground = tuple(
        size for size, holds in zip(values.shape, axes, strict=True) if holds == GROUND
    )
    polar = described["polar"]
    return Crop(
        instrument=described["instrument"],
        identifier=described["identifier"],
        measurement=described["measurement"],
        values=values,
        axes=axes,
        dims={name: tuple(held) for name, held in described["dims"].items()},
        valid=_mask(arrays.pop(VALID, None), ground),
        inside=_mask(arrays.pop(INSIDE, None), ground),
        north=arrays.pop(NORTH),
        east=arrays.pop(EAST),
        separable=described["separable"],
        position_units=described["position_units"],
        polar=None if polar is None else tuple(polar),
        centre_lon=described["centre_lon"],
        centre_lat=described["centre_lat"],
        beside=arrays,
        label=described["label"],
    )


def _native(values: np.ndarray) -> np.ndarray:
    """Return one array in the byte order the machine reads.

    Args:
        values: The values as they were stored, which a build made before the
            writer settled that order leaves most significant byte first.

    Returns:
        values: The same values in the machine's own order, which is the only
            order a tensor takes them in.
    """
    return values.astype(values.dtype.newbyteorder("="), copy=False)


def _mask(held: np.ndarray | None, ground: tuple[int, ...]) -> np.ndarray:
    """Return one mask over the ground, standing everywhere where none was stored.

    Args:
        held: The mask the crop stored, and None where it stored none because
            the mask marked every sample.
        ground: How many samples each ground axis holds, in the order they run.

    Returns:
        mask: The mask, over the ground axes alone.
    """
    return np.ones(ground, dtype=bool) if held is None else held
