"""One patch of one observation, what an embedding places it by, and the layout."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Patch:
    """One patch cut from one observation, and what says where and what it is.

    Attributes:
        values: The patch's own values, in the observation's axis order.
        measured: Whether each sample is a measurement, broadcasting over the values.
        condition: What the model reads beside the values, flattened, zero where
            unmeasured: the cluttergram for SHARAD, else nothing.
        acquisition: The acquisition angles of its observation the model reads, in
            degrees, NaN where the index holds none.
        north_m: How far north of the tile centre the patch centre sits, in metres.
        east_m: How far east of it, in metres.
        delay: Which row of the radargram window the patch centre sounds at, its
            own cut for a sounder and the areoid row for the rest.
        cell: Where the patch sits on its observation's grid of patches, per axis.
    """

    values: np.ndarray
    measured: np.ndarray
    condition: np.ndarray
    acquisition: np.ndarray
    north_m: float
    east_m: float
    delay: float
    cell: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PatchLayout:
    """How each instrument read is cut into patches, and the shape the model reads.

    Attributes:
        sizes: How far a patch of each instrument runs along each axis it is cut on,
            its keys the instruments read.
        shapes: The shape of one patch of each instrument as the model reads it.
    """

    sizes: Mapping[str, Mapping[str, int]]
    shapes: Mapping[str, tuple[int, ...]]

    def only(self, instruments: Iterable[str]) -> PatchLayout:
        """Return the layout of some of the instruments alone.

        Args:
            instruments: The instruments kept.

        Returns:
            layout: Their sizes and shapes alone.
        """
        kept = list(instruments)
        return PatchLayout(
            {name: self.sizes[name] for name in kept},
            {name: self.shapes[name] for name in kept},
        )
