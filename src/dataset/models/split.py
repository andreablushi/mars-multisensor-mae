"""One split of the dataset, each tile read as the patches of each instrument."""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from building.common.layout import Axis
from building.metadata.observation import ObservationMetadata
from torch.utils.data import Dataset

from dataset.models.patch import Patch
from dataset.patches import read_surface_delays, read_tile_patches

if TYPE_CHECKING:
    from dataset.store import DatasetBuild

READY = "ready"


class DatasetSplit(Dataset):
    """Every tile of one split, each read as the patches of each instrument.

    Attributes:
        build: The build the tiles are read from.
        tiles: The index rows of each sensor of each tile, keyed by tile.
        identities: The tiles, in the order the split holds them.
        axes: What each axis of each instrument's values holds.
        sizes: How far a patch of each sensor runs along each axis it is cut on.
        pool: How many ground samples of a patch each instrument averages into one.
        shapes: The shape of one patch of each instrument as the model reads it.
        delay: The instrument whose rows give every surface patch its delay.
    """

    def __init__(
        self,
        build: DatasetBuild,
        tiles: Mapping[str, dict[str, list[ObservationMetadata]]],
        axes: Mapping[str, tuple[str, ...]],
        sizes: Mapping[str, Mapping[str, int]],
        pool: Mapping[str, int],
        shapes: Mapping[str, tuple[int, ...]],
        delay: str,
    ) -> None:
        """Keep what every read needs.

        Args:
            build: The build the tiles are read from.
            tiles: The index rows of each sensor of each tile, keyed by tile.
            axes: What each axis of each instrument's values holds.
            sizes: How far a patch of each sensor runs along each axis it is cut on.
            pool: How many ground samples of a patch each instrument averages into one.
            shapes: The shape of one patch of each instrument as the model reads it.
            delay: The instrument whose rows give every surface patch its delay.
        """
        self.build = build
        self.tiles = tiles
        self.identities = list(tiles)
        self.axes = axes
        self.sizes = sizes
        self.pool = pool
        self.shapes = shapes
        self.delay = delay

    def __len__(self) -> int:
        """Return how many reads the split holds.

        Returns:
            count: One per tile.
        """
        return len(self.identities)

    def __getitem__(self, index: int) -> tuple[dict[str, dict[str, np.ndarray]], str]:
        """Return what one read of a tile holds, and whose it is.

        Args:
            index: Which read of the split.

        Returns:
            sample: Each sensor's patches: values, valid and position.
            identity: The tile the read belongs to.
        """
        identity = self.identities[index]
        held = self.build.root / READY / f"{identity}.npz"
        data = held.read_bytes() if held.is_file() else self.read_ready_tile(identity)
        sample = {}
        with np.load(io.BytesIO(data)) as arrays:
            for packed in arrays.files:
                name, key = packed.split("/")
                sample.setdefault(name, {})[key] = arrays[packed]
        return sample, identity

    def read_ready_tile(self, identity: str) -> bytes:
        """Return one tile cut, scaled and packed, kept under the root on the way.

        Args:
            identity: The tile to read.

        Returns:
            data: The packed arrays of every sensor's patches.
        """
        rows = self.tiles[identity]
        delays = read_surface_delays(self.build, rows[self.delay])
        read = read_tile_patches(rows, self.build, self.sizes, self.pool, delays)
        packed = io.BytesIO()
        np.savez_compressed(
            packed,
            **{
                f"{name}/{key}": array
                for name, drawn in read.items()
                for key, array in patch_arrays(
                    drawn, self.shapes[name], self.axes[name]
                ).items()
            },
        )
        self.build.keep(f"{READY}/{identity}.npz", packed.getvalue())
        return packed.getvalue()


def patch_arrays(
    patches: Sequence[Patch],
    shape: Sequence[int],
    axes: Sequence[str],
) -> dict[str, np.ndarray]:
    """Return one instrument's read patches as the arrays a model is handed.

    Args:
        patches: The patches, all of one instrument.
        shape: The shape of one patch of it as the model reads it.
        axes: What each axis of its values holds.

    Returns:
        arrays: The patches under "values", "valid" and "position".
    """
    valid_shape = tuple(
        held if holds in (Axis.GROUND, Axis.WAVELENGTH) else 1
        for held, holds in zip(shape, axes, strict=True)
    )
    return {
        "values": stacked([one.values for one in patches], tuple(shape), np.float32),
        "valid": stacked([one.valid for one in patches], valid_shape, bool),
        "position": stacked(
            [
                np.array(
                    [
                        one.east_m,
                        one.north_m,
                        one.delay,
                        one.east_span_m,
                        one.north_span_m,
                        one.delay_span,
                    ],
                    np.float32,
                )
                for one in patches
            ],
            (6,),
            np.float32,
        ),
    }


def stacked(
    arrays: Sequence[np.ndarray], shape: tuple[int, ...], dtype: np.dtype | type
) -> np.ndarray:
    """Return same-shaped arrays as one array, empty where none were given.

    Args:
        arrays: The arrays, every one of that shape.
        shape: Their shape, which an empty list cannot say.
        dtype: What they hold, which an empty list cannot say either.

    Returns:
        stacked: The arrays along a new first axis. (K, *shape)
    """
    if not arrays:
        return np.zeros((0, *shape), dtype)
    return np.stack(arrays).astype(dtype, copy=False)
