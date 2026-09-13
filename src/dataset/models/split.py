"""One split of the dataset, each feature read as every patch of each instrument."""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
from building.common.layout import GROUND
from building.metadata.observation import ObservationMetadata
from torch.utils.data import Dataset

from dataset.models.patch import Patch
from dataset.patches import read_patches

if TYPE_CHECKING:
    from dataset.store import DatasetBuild


def stacked(
    arrays: list[np.ndarray], shape: tuple[int, ...], dtype: np.dtype | type
) -> np.ndarray:
    """Return a list of same-shaped arrays as one array, empty where none were given.

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


class DatasetSplit(Dataset):
    """Every feature of one split, each read as every patch of each instrument.

    Attributes:
        build: The build the features are read from.
        features: The index rows of each instrument of each feature of the
            split, keyed by what tells the feature apart.
        identities: The features, in the order the split holds them.
        axes: What each axis of each instrument's values holds.
        statistics: What each instrument's values run to over the training
            split, keyed as ODE names it.
        sizes: How far a patch of each instrument runs along an axis it is cut
            on, keyed as ODE names it.
        shapes: The shape of one patch of each instrument as the model reads it.
        elevation: The instrument whose values give every surface patch its
            height.
        budget: How many patches of each instrument one read draws at most.
        seed: The number that fixes every read's draw, or None for a split that
            draws anew each time.
    """

    def __init__(
        self,
        build: DatasetBuild,
        features: Mapping[tuple[str, str], dict[str, list[ObservationMetadata]]],
        axes: Mapping[str, tuple[str, ...]],
        statistics: Mapping[str, dict[str, float]],
        sizes: Mapping[str, int],
        shapes: Mapping[str, tuple[int, ...]],
        elevation: str,
        budget: int,
        seed: int | None,
    ) -> None:
        """Keep what every read needs, and check every instrument can be normalised.

        Args:
            build: The build the features are read from.
            features: The index rows of each instrument of each feature of the
                split, keyed by what tells the feature apart.
            axes: What each axis of each instrument's values holds.
            statistics: What each instrument's values run to over the training
                split, keyed as ODE names it.
            sizes: How far a patch of each instrument runs along an axis it is
                cut on, keyed as ODE names it.
            shapes: The shape of one patch of each instrument as the model
                reads it.
            elevation: The instrument whose values give every surface patch its
                height.
            budget: How many patches of each instrument one read draws at most.
            seed: The number that fixes every read's draw, or None for a split
                that draws anew each time.

        Raises:
            ValueError: When an instrument the model reads has no finite
                statistics to normalise its patches by.
        """
        self.build = build
        self.features = features
        self.identities = list(features)
        self.axes = axes
        self.statistics = statistics
        self.sizes = sizes
        self.shapes = shapes
        self.elevation = elevation
        self.budget = budget
        self.seed = seed
        for name in sizes:
            if name not in statistics:
                raise ValueError(f"{name} has no finite statistics to normalise by")

    def __len__(self) -> int:
        """Return how many features the split holds.

        Returns:
            count: The number of features.
        """
        return len(self.identities)

    def __getitem__(self, index: int) -> tuple[dict[str, dict[str, np.ndarray]], str]:
        """Return what one feature was drawn as, and its class.

        Args:
            index: Which feature of the split.

        Returns:
            sample: For each instrument the model reads, its normalised patches
                under "values" (K, *P), whether each sample of them is a
                measurement under "valid" (K, *P'), and where each sits and how
                far it reaches, in metres, under "position" (K, 6), K being
                what the budget drew of the feature, which may be none. A split
                holding a seed draws the same patches of a feature every time,
                so its loss is the last pass's to beat; one holding none draws
                anew, so a run reads every patch of a feature over its epochs.
            feature_class: The class of the feature, as ODE names it.

        Raises:
            ValueError: When the feature has no elevation to stand on.
        """
        identity = self.identities[index]
        rows = self.features[identity]
        held = rows.get(self.elevation)
        if not held:
            raise ValueError(f"{identity} has no {self.elevation} to stand on")
        heights = self.build.read_heights(held)
        draw = random.Random(None if self.seed is None else f"{self.seed}/{identity}")
        sample = {}
        for name, size in self.sizes.items():
            drawn = read_patches(
                rows.get(name, []), self.build, size, heights, self.budget, draw
            )
            shape = self.shapes[name]
            valid_shape = tuple(
                held if holds == GROUND else 1
                for held, holds in zip(shape, self.axes[name], strict=True)
            )
            sample[name] = {
                "values": stacked(
                    [self.normalised(patch, valid_shape) for patch in drawn],
                    shape,
                    np.float32,
                ),
                "valid": stacked(
                    [patch.valid.reshape(valid_shape) for patch in drawn],
                    valid_shape,
                    bool,
                ),
                "position": stacked(
                    [
                        np.array(
                            [
                                patch.east_m,
                                patch.north_m,
                                patch.height_m,
                                patch.east_span_m,
                                patch.north_span_m,
                                patch.height_span_m,
                            ],
                            np.float32,
                        )
                        for patch in drawn
                    ],
                    (6,),
                    np.float32,
                ),
            }
        return sample, identity[0]

    def normalised(self, patch: Patch, valid_shape: tuple[int, ...]) -> np.ndarray:
        """Return one patch's values centred and scaled by its instrument's moments.

        Args:
            patch: The patch.
            valid_shape: The shape its measurement mask takes to broadcast over
                the values.

        Returns:
            values: The values less their mean over their deviation, the
                instrument's own over the training split, one number for an
                instrument measuring one thing and one per band for a spectral
                one, and zero where the sample is no measurement. An encoder
                reads every sample of a patch, mask or no mask, so a sample a
                sounder wrote as nan is read as the instrument's average
                instead. (*P)
        """
        held = self.statistics[patch.instrument]
        values = patch.values.astype(np.float32)  # (*P)
        scaled = (values - held["mean"]) / np.maximum(held["deviation"], 1e-6)  # (*P)
        return np.where(patch.valid.reshape(valid_shape), scaled, 0.0)  # (*P)
