"""One split of the dataset, each tile read as the patches of each instrument."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from building.common.layout import GROUND, WAVELENGTH
from building.metadata.observation import ObservationMetadata
from torch.utils.data import Dataset

from dataset.models.patch import Patch
from dataset.patches import channel_axis, draw_tile_patches, read_tile_patches

if TYPE_CHECKING:
    from dataset.store import DatasetBuild


class DatasetSplit(Dataset):
    """Every tile of one split, each read as the patches of each instrument.

    Attributes:
        build: The build the tiles are read from.
        tiles: The index rows of each sensor of each tile, keyed by identity.
        identities: The tiles, in the order the split holds them.
        axes: What each axis of each instrument's values holds.
        statistics: What each sensor's values run to over the training split.
        sizes: How far a patch of each sensor runs along an axis it is cut on.
        shapes: The shape of one patch of each instrument as the model reads it.
        wavelengths: What each band of each spectral sensor is centred on, in nm.
        elevation: The instrument whose values give every surface patch its height.
        budget: How many patches of each instrument one draw takes at most.
        overlap: The share of each other sensor's patches over the anchor's ground.
        seed: What fixes every read's draw, or None to draw anew each time.
        ceiling: How many patches one whole read hands back, or None to draw.
        kept: What each seeded draw held, since it hands back the same every read.
    """

    def __init__(
        self,
        build: DatasetBuild,
        tiles: Mapping[str, dict[str, list[ObservationMetadata]]],
        axes: Mapping[str, tuple[str, ...]],
        statistics: Mapping[str, dict[str, np.ndarray]],
        sizes: Mapping[str, int],
        shapes: Mapping[str, tuple[int, ...]],
        wavelengths: Mapping[str, tuple[float, ...]],
        elevation: str,
        budget: int,
        overlap: float,
        seed: int | None,
        ceiling: int | None = None,
    ) -> None:
        """Keep what every read needs, and check every instrument can be normalised.

        Args:
            build: The build the tiles are read from.
            tiles: The index rows of each sensor of each tile, keyed by identity.
            axes: What each axis of each instrument's values holds.
            statistics: What each sensor's values run to over the training split.
            sizes: How far a patch of each sensor runs along an axis it is cut on.
            shapes: The shape of one patch of each instrument as the model reads it.
            wavelengths: What each band of each spectral sensor is centred on, in nm.
            elevation: The instrument whose values give every surface patch its height.
            budget: How many patches of each instrument one draw takes at most.
            overlap: The share of each other sensor's patches over the anchor's ground.
            seed: What fixes every read's draw, or None to draw anew each time.
            ceiling: How many patches one whole read hands back, or None to draw.

        Raises:
            ValueError: When a sensor the model reads has no statistics to scale by.
        """
        self.build = build
        self.tiles = tiles
        self.identities = list(tiles)
        self.axes = axes
        self.statistics = statistics
        self.sizes = sizes
        self.shapes = shapes
        self.wavelengths = wavelengths
        self.elevation = elevation
        self.budget = budget
        self.overlap = overlap
        self.seed = seed
        self.ceiling = ceiling
        self.kept: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        for name in sizes:
            if name not in statistics:
                raise ValueError(f"{name} has no finite statistics to normalise by")

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
            sample: Each sensor's patches: values, valid, channels and position.
            identity: The tile the read belongs to, its class and its name.

        Raises:
            ValueError: When the tile has no elevation to stand on.
        """
        identity = self.identities[index]
        if identity in self.kept:
            return self.kept[identity], identity
        rows = self.tiles[identity]
        held = rows.get(self.elevation)
        if not held:
            raise ValueError(f"{identity} has no {self.elevation} to stand on")
        heights = self.build.read_heights(held)
        if self.ceiling is None:
            draw = random.Random(
                None if self.seed is None else f"{self.seed}/{identity}"
            )
            read = draw_tile_patches(
                rows,
                self.build,
                self.sizes,
                self.wavelengths,
                heights,
                self.budget,
                self.overlap,
                draw,
            )
        else:
            read = read_tile_patches(
                rows,
                self.build,
                self.sizes,
                self.wavelengths,
                heights,
                self.ceiling,
            )
        sample = {
            name: patch_arrays(
                drawn, self.shapes[name], self.axes[name], self.statistics[name]
            )
            for name, drawn in read.items()
        }
        if self.seed is not None and self.ceiling is None:
            self.kept[identity] = sample
        return sample, identity


def patch_arrays(
    patches: Sequence[Patch],
    shape: Sequence[int],
    axes: Sequence[str],
    statistics: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return one instrument's read patches as the arrays a model is handed.

    Args:
        patches: The patches, all of one instrument.
        shape: The shape of one patch of it as the model reads it.
        axes: What each axis of its values holds.
        statistics: What its values run to over the training split.

    Returns:
        arrays: The patches under "values", "valid", "channels" and "position".
    """
    valid_shape = tuple(
        held if holds in (GROUND, WAVELENGTH) else 1
        for held, holds in zip(shape, axes, strict=True)
    )
    at = channel_axis(axes)
    channels = shape[at] if at is not None else 1
    scaled = [
        np.where(
            one.valid,
            (one.values.astype(np.float32) - statistics["mean"])
            / np.maximum(statistics["deviation"], 1e-6),
            0.0,
        )
        for one in patches
    ]
    return {
        "values": stacked(scaled, tuple(shape), np.float32),
        "valid": stacked([one.valid for one in patches], valid_shape, bool),
        "channels": stacked([one.channels for one in patches], (channels,), np.float32),
        "position": stacked(
            [
                np.array(
                    [
                        one.east_m,
                        one.north_m,
                        one.height_m,
                        one.east_span_m,
                        one.north_span_m,
                        one.height_span_m,
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
    return np.stack(arrays)
