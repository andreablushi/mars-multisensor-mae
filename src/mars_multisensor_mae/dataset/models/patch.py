"""One patch of one observation, a tile's patches as arrays, and the layout."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from mars_multisensor_dataset.building.common.layout import Axis
from mars_multisensor_dataset.building.configs import sharad
from mars_multisensor_dataset.building.metadata.observation import ObservationMetadata


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


class InstrumentPatches(NamedTuple):
    """One instrument's patches over one tile, as cached and read by a model.

    Attributes:
        values: The normalised patches. (N, *P)
        measured: Whether each sample is a measurement, broadcasting over the values.
            (N, *P')
        condition: What each patch is conditioned on, flattened, or nothing. (N, C)
        acquisition: The acquisition angles of each patch's observation, in degrees,
            or nothing. (N, A)
        position: The patch centre, east and north in metres, delay in rows. (N, 3)
    """

    values: np.ndarray
    measured: np.ndarray
    condition: np.ndarray
    acquisition: np.ndarray
    position: np.ndarray

    @classmethod
    def stacked(
        cls,
        patches: Sequence[Patch],
        name: str,
        shape: Sequence[int],
        axes: Sequence[str],
        acquisition: Sequence[str],
    ) -> InstrumentPatches:
        """Return some patches of one instrument stacked into its arrays.

        Args:
            patches: The patches, all of one instrument.
            name: The instrument, keyed as ODE names it.
            shape: The shape of one patch of it as the model reads it.
            axes: What each axis of its values holds.
            acquisition: The acquisition angles each patch carries, as the index
                names them.

        Returns:
            patches: The patches, one row each, in their order.
        """
        measured_shape = tuple(
            held if holds == Axis.GROUND else 1
            for held, holds in zip(shape, axes, strict=True)
        )
        width = math.prod(shape) if name == sharad.LAYOUT.instrument else 0
        return cls(
            values=np.array([one.values for one in patches], np.float32).reshape(
                -1, *shape
            ),
            measured=np.array([one.measured for one in patches], bool).reshape(
                -1, *measured_shape
            ),
            condition=np.array([one.condition for one in patches], np.float32).reshape(
                len(patches), width
            ),
            acquisition=np.array(
                [one.acquisition for one in patches], np.float32
            ).reshape(len(patches), len(acquisition)),
            position=np.array(
                [[one.east_m, one.north_m, one.delay] for one in patches], np.float32
            ).reshape(-1, 3),
        )


@dataclass(frozen=True, slots=True)
class TileSample:
    """Every instrument's patches over one tile or one crop of it.

    Attributes:
        instruments: Each instrument's patches, keyed as ODE names it.
    """

    instruments: dict[str, InstrumentPatches]

    @property
    def ground_spots(self) -> np.ndarray:
        """Return every ground spot a patch stands on, each once, sorted.

        Returns:
            spots: East and north of each spot in metres, whether a sounder's delay
                is cut or not. (G, 2)
        """
        return np.unique(
            np.concatenate(
                [held.position[:, :2] for held in self.instruments.values()]
            ),
            axis=0,
        )

    @property
    def measured_ground_spots(self) -> np.ndarray:
        """Return the ground spot of every patch holding a measurement, any instrument.

        Returns:
            spots: East and north of each such patch in metres, as many as there are
                such patches. (S, 2)
        """
        spots = []
        for held in self.instruments.values():
            present = held.measured.any(tuple(range(1, held.measured.ndim)))  # (N,)
            spots.append(held.position[present, :2])
        return np.concatenate(spots)

    def budget_cut(self, budget: Mapping[str, int], centre: np.ndarray) -> TileSample:
        """Return the sample cut to the budget around one ground spot.

        Args:
            budget: How many patches of each instrument are kept.
            centre: East and north of the spot every instrument is cut around, in
                metres. (2,)

        Returns:
            cut: Each instrument's patches nearest the spot alone, ties in their order.
        """
        cut = {}
        for name, held in self.instruments.items():
            apart = np.linalg.norm(held.position[:, :2] - centre, axis=1)
            kept = np.argsort(apart, kind="stable")[: budget[name]]
            cut[name] = InstrumentPatches(*(array[kept] for array in held))
        return TileSample(cut)


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

    @classmethod
    def from_index(
        cls,
        by_tile: Mapping[str, Mapping[str, Sequence[ObservationMetadata]]],
        sizes: Mapping[str, Mapping[str, int]],
    ) -> PatchLayout:
        """Return the layout the index settles, an axis a size omits taken whole.

        Args:
            by_tile: The index rows of each instrument of each tile.
            sizes: How far a patch of each instrument runs along each axis it is cut
                on.

        Returns:
            layout: The sizes, and the shape of one patch of each instrument.
        """
        shapes = {}
        for name, size in sizes.items():
            row = next(rows[name][0] for rows in by_tile.values() if name in rows)
            shapes[name] = tuple(
                size.get(holds, length)
                for length, holds in zip(row.shape, row.axes, strict=True)
            )
        return cls(sizes, shapes)

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
