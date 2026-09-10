"""Drawing every feature of a split as a few patches of each instrument, in batches."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from building.common.layout import GROUND, WAVELENGTH
from building.metadata.observation import ObservationMetadata
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from architecture.tokens import Tokens
from config.schema import Config
from dataset.models.patch import Patch
from dataset.patches import cut_patch, patch_counts, read_heights
from dataset.store import DatasetBuild

TRAINING_SPLIT = "train"
VALIDATION_SPLIT = "validation"


def split_loaders(
    build: DatasetBuild, config: Config
) -> tuple[DataLoader, DataLoader, dict[str, tuple[int, ...]]]:
    """Return the training and validation splits in batches, and every token shape.

    Args:
        build: The build the features are read from.
        config: What the run reads, how it draws it, and how it batches it.

    Returns:
        training: The training split, every feature drawn afresh at every read.
        validation: The validation split, every feature drawn the same at
            every read.
        shapes: The shape of one patch of each instrument as the model reads
            it, keyed as ODE names it.
    """
    by_feature = build.read_observation_metadata_by_feature()
    splits = build.split_features(config.dataset.split, config.dataset.seed)
    statistics = build.compute_stats(set(splits[TRAINING_SPLIT]))
    axes = {
        name: tuple(held[0].axes)
        for feature in by_feature.values()
        for name, held in feature.items()
    }

    def loader(split: str, seed: int | None) -> DataLoader:
        """Return one split in batches, its features drawn afresh or fixed."""
        features = {identity: by_feature[identity] for identity in splits[split]}
        held = FeatureDataset(build, features, axes, config, statistics, seed)
        return DataLoader(
            held,
            batch_size=config.training.batch_size,
            shuffle=seed is None,
            num_workers=config.training.workers,
            collate_fn=collate,
        )

    training = loader(TRAINING_SPLIT, None)
    validation = loader(VALIDATION_SPLIT, config.dataset.seed)
    return training, validation, token_shapes(axes, config)


def patch_sizes(config: Config) -> dict[str, int]:
    """Return how far a patch of each instrument the model reads runs along an axis.

    Args:
        config: Which instruments the model reads, and how far a patch runs
            along every axis it is cut on, by instrument, and under "default"
            for every instrument unnamed.

    Returns:
        sizes: One length per instrument, keyed as ODE names it.
    """
    held = config.dataset.patchsize
    return {name: held.get(name, held["default"]) for name in config.model.instruments}


def token_shapes(
    axes: dict[str, tuple[str, ...]], config: Config
) -> dict[str, tuple[int, ...]]:
    """Return the shape of one patch of each instrument as the model reads it.

    Args:
        axes: What each axis of each instrument's values holds.
        config: How far a patch runs, and how many bands a spectral one keeps.

    Returns:
        shapes: One shape per instrument the model reads, keyed as ODE names it.
    """
    sizes = patch_sizes(config)
    return {
        name: tuple(
            config.model.bands if holds == WAVELENGTH else sizes[name]
            for holds in axes[name]
        )
        for name in config.model.instruments
    }


def draw_patches(
    observations: Sequence[ObservationMetadata],
    build: DatasetBuild,
    patchsize: int,
    count: int,
    heights: np.ndarray,
    rng: np.random.Generator,
) -> list[tuple[ObservationMetadata, Patch]]:
    """Return patches of one instrument drawn at random over one feature.

    Args:
        observations: The feature's index rows of that one instrument.
        build: The published build the observations are read from.
        patchsize: How far a patch of that instrument runs along every axis it
            is cut on.
        count: How many to draw, or every one where the feature holds fewer.
        heights: Where the ground stands over the feature. (N, 3)
        rng: What fixes the draw.

    Returns:
        drawn: Each patch beside the row it was cut from, every whole patch of
            every observation equally likely and none twice. An observation is
            read only when a patch of it was drawn. Empty where none holds one.
    """
    totals = [
        math.prod(patch_counts(one.shape, one.axes, patchsize)) for one in observations
    ]
    edges = np.cumsum([0, *totals])
    chosen = np.sort(rng.choice(edges[-1], size=min(count, edges[-1]), replace=False))
    drawn = []
    for at, record in enumerate(observations):
        taken = chosen[(chosen >= edges[at]) & (chosen < edges[at + 1])] - edges[at]
        if taken.size == 0:
            continue
        observation = build.read_observation(record.path)
        drawn.extend(
            (record, cut_patch(observation, record, int(one), patchsize, heights))
            for one in taken
        )
    return drawn


def draw_hidden_patches(
    count: int, ratio: float, rng: np.random.Generator
) -> np.ndarray:
    """Return which of one instrument's patches of one feature to hide.

    Args:
        count: How many patches the feature was drawn with.
        ratio: The share of them to hide from the instrument's encoder.
        rng: What fixes the draw.

    Returns:
        hidden: The patches drawn, at random, that share of them rounded down.
            (K,)
    """
    hidden = np.zeros(count, dtype=bool)  # (K,)
    hidden[rng.choice(count, size=int(count * ratio), replace=False)] = True
    return hidden


def resample_bands(values: np.ndarray, axis: int, bands: int) -> np.ndarray:
    """Return the values with their wavelength axis resampled to a fixed count.

    Args:
        values: One patch's values. (*P)
        axis: Which axis runs over wavelength.
        bands: How many bands to keep.

    Returns:
        resampled: The values read linearly between neighbouring bands at that
            many evenly spaced points along the axis. (*P')
    """
    held = values.shape[axis]
    at = np.linspace(0, held - 1, bands)  # (bands,)
    low = np.floor(at).astype(int)  # (bands,)
    high = np.minimum(low + 1, held - 1)  # (bands,)
    spread = [1] * values.ndim
    spread[axis] = bands
    weight = (at - low).astype(values.dtype).reshape(spread)  # (1, .., bands, .., 1)
    return (
        np.take(values, low, axis) * (1 - weight) + np.take(values, high, axis) * weight
    )


def stacked(arrays: list[np.ndarray], shape: tuple[int, ...]) -> Tensor:
    """Return a list of same-shaped arrays as one tensor, empty where none were given.

    Args:
        arrays: The arrays, every one of that shape.
        shape: Their shape, which an empty list cannot say.

    Returns:
        stacked: The arrays along a new first axis. (K, *shape)
    """
    if not arrays:
        return torch.zeros((0, *shape))
    return torch.as_tensor(np.stack(arrays))


class FeatureDataset(Dataset):
    """Every feature of one split, each read as a few patches of each instrument.

    Attributes:
        build: The build the features are read from.
        features: The index rows of each instrument of each feature of the
            split, keyed by what tells the feature apart.
        identities: The features, in the order the split holds them.
        axes: What each axis of each instrument's values holds.
        config: What the run reads and what the model draws.
        statistics: What each instrument's values run to over the training
            split, keyed as ODE names it.
        seed: What fixes the draw of every feature, or None for a fresh draw at
            every read.
        patchsize: How far a patch of each instrument runs along an axis it is
            cut on.
        shapes: The shape of one patch of each instrument as the model reads it.
    """

    def __init__(
        self,
        build: DatasetBuild,
        features: Mapping[tuple[str, str], dict[str, list[ObservationMetadata]]],
        axes: dict[str, tuple[str, ...]],
        config: Config,
        statistics: dict[str, dict[str, float]],
        seed: int | None = None,
    ) -> None:
        """Keep what every read needs, and check every instrument can be normalised.

        Args:
            build: The build the features are read from.
            features: The index rows of each instrument of each feature of the
                split, keyed by what tells the feature apart.
            axes: What each axis of each instrument's values holds.
            config: What the run reads and what the model draws.
            statistics: What each instrument's values run to over the training
                split, keyed as ODE names it.
            seed: What fixes the draw of every feature, or None for a fresh
                draw at every read.
        """
        self.build = build
        self.features = features
        self.identities = list(features)
        self.axes = axes
        self.config = config
        self.statistics = statistics
        self.seed = seed
        self.patchsize = patch_sizes(config)
        self.shapes = token_shapes(axes, config)
        for name in config.model.instruments:
            if WAVELENGTH not in axes[name] and name not in statistics:
                raise ValueError(f"{name} has no finite statistics to normalise by")

    def __len__(self) -> int:
        """Return how many features the split holds.

        Returns:
            count: The number of features.
        """
        return len(self.identities)

    def __getitem__(self, index: int) -> tuple[dict[str, dict[str, Tensor]], str]:
        """Return one feature's patches of each instrument as tensors, and its class.

        Args:
            index: Which feature of the split.

        Returns:
            sample: For each instrument the model reads, its normalised patches
                under "values" (K, *P), whether each sample of them is a
                measurement under "valid" (K, *P'), where each sits in metres
                under "position" (K, 3), and which of them its encoder may read
                under "visible" (K,), K being how many the feature was drawn
                with, which may be none.
            feature_class: The class of the feature, as ODE names it.
        """
        identity = self.identities[index]
        rows = self.features[identity]
        rng = np.random.default_rng(None if self.seed is None else (self.seed, index))
        held = rows.get(self.config.model.elevation)
        if not held:
            raise ValueError(
                f"{identity} has no {self.config.model.elevation} to stand on"
            )
        heights = read_heights(held, self.build)
        sample = {}
        for name in self.config.model.instruments:
            drawn = draw_patches(
                rows.get(name, []),
                self.build,
                self.patchsize[name],
                self.config.model.patches,
                heights,
                rng,
            )
            shape = self.shapes[name]
            valid_shape = tuple(
                size if holds == GROUND else 1
                for size, holds in zip(shape, self.axes[name], strict=True)
            )
            values, valid, position = [], [], []
            for record, patch in drawn:
                if not patch.valid.any():
                    continue
                values.append(self.normalised(record, patch))
                valid.append(patch.valid.reshape(valid_shape))
                position.append(
                    np.array([patch.east_m, patch.north_m, patch.height_m], np.float32)
                )
            hidden = draw_hidden_patches(
                len(values), self.config.model.mask_ratio, rng
            )  # (K,)
            sample[name] = {
                "values": stacked(values, shape),
                "valid": stacked(valid, valid_shape),
                "position": stacked(position, (3,)),
                "visible": torch.as_tensor(~hidden),
            }
        return sample, identity[0]

    def normalised(self, record: ObservationMetadata, patch: Patch) -> np.ndarray:
        """Return one patch's values centred and scaled, a spectral one on fixed bands.

        Args:
            record: The row of the observation the patch was cut from, which
                carries the moments of every band of a spectral one.
            patch: The patch.

        Returns:
            values: The values less their mean over their deviation, the
                instrument's own over the training split, or each band's own
                over its observation, then resampled to the fixed band count.
                (*P)
        """
        values = patch.values.astype(np.float32)  # (*P)
        if WAVELENGTH not in patch.axes:
            held = self.statistics[patch.instrument]
            return (values - held["mean"]) / max(held["deviation"], 1e-6)
        axis = patch.axes.index(WAVELENGTH)
        spread = [1] * values.ndim
        spread[axis] = -1
        mean = np.asarray(record.band_mean, dtype=np.float32).reshape(spread)
        deviation = np.asarray(record.band_std, dtype=np.float32).reshape(spread)
        values = (values - mean) / np.maximum(deviation, 1e-6)  # (*P)
        return resample_bands(values, axis, self.config.model.bands)


def collate(
    samples: list[tuple[dict[str, dict[str, Tensor]], str]],
) -> tuple[dict[str, Tokens], list[str]]:
    """Return one batch of every instrument's tokens, and the features' classes.

    Args:
        samples: What the dataset read of each feature of the batch.

    Returns:
        batch: Each instrument's patches padded to the most any feature of
            the batch holds, keyed as ODE names it.
        classes: The class of each feature, in the batch's order.
    """
    batch = {}
    for name in samples[0][0]:
        held = [sample[name] for sample, _ in samples]
        counts = torch.tensor([one["values"].shape[0] for one in held])  # (B,)
        slots = torch.arange(int(counts.max()))  # (K,)
        padded = {
            key: pad_sequence([one[key] for one in held], batch_first=True)
            for key in held[0]
        }
        batch[name] = Tokens(**padded, present=slots.unsqueeze(0) < counts.unsqueeze(1))
    return batch, [feature_class for _, feature_class in samples]
