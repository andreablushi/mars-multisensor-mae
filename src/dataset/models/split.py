"""One split of the dataset, each feature read as every patch of each instrument."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from building.metadata.observation import ObservationMetadata
from torch.utils.data import Dataset

from dataset.models.patch import Patch
from dataset.patches import (
    cut_patch,
    every_patch_plan,
    patch_arrays,
    read_feature_patches,
)

if TYPE_CHECKING:
    from dataset.store import DatasetBuild


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
        wavelengths: What each band of each spectral instrument is centred on,
            in nanometres, keyed as ODE names it.
        elevation: The instrument whose values give every surface patch its
            height.
        budget: How many patches of each instrument one read draws at most.
        overlap: The share of every other instrument's patches that must reach
            ground the anchor instrument's patches reach.
        seed: The number that fixes every read's draw, or None for a split that
            draws anew each time.
        chunk: How many patches of one instrument one read hands back, reading
            the split whole a chunk at a time, or None to draw once per feature.
        reads: Which feature, and which of its chunks, each read stands for.
        opened: The observations last read, which the next chunk of one reuses.
    """

    def __init__(
        self,
        build: DatasetBuild,
        features: Mapping[tuple[str, str], dict[str, list[ObservationMetadata]]],
        axes: Mapping[str, tuple[str, ...]],
        statistics: Mapping[str, dict[str, float]],
        sizes: Mapping[str, int],
        shapes: Mapping[str, tuple[int, ...]],
        wavelengths: Mapping[str, tuple[float, ...]],
        elevation: str,
        budget: int,
        overlap: float,
        seed: int | None,
        chunk: int | None = None,
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
            wavelengths: What each band of each spectral instrument is centred
                on, in nanometres, keyed as ODE names it.
            elevation: The instrument whose values give every surface patch its
                height.
            budget: How many patches of each instrument one read draws at most.
            overlap: The share of every other instrument's patches that must
                reach ground the anchor instrument's patches reach.
            seed: The number that fixes every read's draw, or None for a split
                that draws anew each time.
            chunk: How many patches of one instrument one read hands back, or
                None to draw `budget` of each once per feature.

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
        self.wavelengths = wavelengths
        self.elevation = elevation
        self.budget = budget
        self.overlap = overlap
        self.seed = seed
        self.chunk = chunk
        self.opened: dict[str, object] = {}
        self.reads = [(at, None) for at in range(len(self.identities))]
        if chunk is not None:
            self.reads = [
                (at, planned)
                for at, identity in enumerate(self.identities)
                for planned in every_patch_plan(features[identity], sizes, chunk)
            ]
        for name in sizes:
            if name not in statistics:
                raise ValueError(f"{name} has no finite statistics to normalise by")

    def __len__(self) -> int:
        """Return how many reads the split holds.

        Returns:
            count: One per feature, or one per chunk where the split is read whole.
        """
        return len(self.reads)

    def __getitem__(
        self, index: int
    ) -> tuple[dict[str, dict[str, np.ndarray]], tuple[str, str]]:
        """Return what one read of a feature holds, and whose it is.

        Args:
            index: Which read of the split.

        Returns:
            sample: For each instrument the model reads, its normalised patches
                under "values" (K, *P), whether each sample of them is a
                measurement under "valid" (K, *P'), what each of its channels
                measures under "channels" (K, C), and where each sits and how
                far it reaches, in metres, under "position" (K, 6), which may
                hold none. A split holding a seed draws the same patches every
                time, so its loss is the last pass's to beat; one holding none
                draws anew, so a run reads every patch over its epochs.
            identity: The feature the read belongs to, its class and its name,
                so the chunks of one feature are told from another's.

        Raises:
            ValueError: When the feature has no elevation to stand on.
        """
        at, planned = self.reads[index]
        identity = self.identities[at]
        rows = self.features[identity]
        held = rows.get(self.elevation)
        if not held:
            raise ValueError(f"{identity} has no {self.elevation} to stand on")
        heights = self.build.read_heights(held)
        if planned is None:
            draw = random.Random(
                None if self.seed is None else f"{self.seed}/{identity}"
            )
            read = read_feature_patches(
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
            read = {
                name: self.cut_planned(name, taken, heights)
                for name, taken in planned.items()
            }
        sample = {
            name: patch_arrays(
                drawn, self.shapes[name], self.axes[name], self.statistics[name]
            )
            for name, drawn in read.items()
        }
        return sample, identity

    def cut_planned(
        self,
        name: str,
        taken: Sequence[tuple[ObservationMetadata, int]],
        heights: np.ndarray,
    ) -> list[Patch]:
        """Return the planned patches of one instrument, cut from their observations.

        Args:
            name: The instrument, as ODE names it.
            taken: Which observation and which patch of it to cut, in order.
            heights: Where the ground stands over the feature. (N, 3)

        Returns:
            patches: The patches, holding none that measured nothing.
        """
        # The chunks of one observation run together, so it is read once for all.
        cut = []
        for record, at in taken:
            if record.path not in self.opened:
                self.opened = {record.path: self.build.read_observation(record.path)}
            patch = cut_patch(
                self.opened[record.path],
                record,
                at,
                self.sizes[name],
                heights,
                self.wavelengths.get(name, ()),
            )
            if patch.valid.any():
                cut.append(patch)
        return cut
