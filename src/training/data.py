"""Reading features as batches of tokens, a few patches of each instrument at a time."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import torch
from building.common.layout import GROUND, WAVELENGTH
from building.metadata.observation import ObservationMetadata
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from architecture.tokens import Tokens
from config.schema import Config
from dataset.elevation import Elevation, patch_elevation, read_elevation
from dataset.models.patch import Patch
from dataset.sample import draw_patches
from dataset.store import DatasetBuild


def rows_by_instrument(
    rows: Sequence[ObservationMetadata],
) -> dict[str, list[ObservationMetadata]]:
    """Return one feature's index rows grouped by the instrument that took them.

    Args:
        rows: The feature's rows, in the order the index holds them.

    Returns:
        grouped: The rows of each instrument, keyed as ODE names it.
    """
    grouped = defaultdict(list)
    for one in rows:
        grouped[one.instrument].append(one)
    return dict(grouped)


def axes_by_instrument(
    rows: Sequence[ObservationMetadata],
) -> dict[str, tuple[str, ...]]:
    """Return what each axis of each instrument's values holds, read off any row of it.

    Args:
        rows: Any rows of the index, at least one per instrument.

    Returns:
        axes: What each axis holds, in the values' own order, keyed as ODE
            names the instrument.
    """
    return {one.instrument: tuple(one.axes) for one in rows}


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
    patchsize = config.dataset.patchsize
    return {
        name: tuple(
            config.model.bands
            if holds == WAVELENGTH
            else patchsize.get(name, patchsize["default"])
            for holds in axes[name]
        )
        for name in config.model.instruments
    }


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
        features: The index rows of each feature of the split.
        axes: What each axis of each instrument's values holds.
        shapes: The shape of one patch of each instrument as the model reads it.
        config: What the run reads and what the model draws.
        statistics: What each instrument's values run to over the training
            split, keyed as ODE names it.
        seed: What fixes the draw of every feature, or None for a fresh draw at
            every read.
    """

    def __init__(
        self,
        build: DatasetBuild,
        features: Sequence[Sequence[ObservationMetadata]],
        axes: dict[str, tuple[str, ...]],
        config: Config,
        statistics: dict[str, dict[str, float]],
        seed: int | None = None,
    ) -> None:
        """Keep what every read needs, and check every instrument can be normalised.

        Args:
            build: The build the features are read from.
            features: The index rows of each feature of the split.
            axes: What each axis of each instrument's values holds.
            config: What the run reads and what the model draws.
            statistics: What each instrument's values run to over the training
                split, keyed as ODE names it.
            seed: What fixes the draw of every feature, or None for a fresh
                draw at every read.
        """
        self.build = build
        self.features = features
        self.axes = axes
        self.shapes = token_shapes(axes, config)
        self.config = config
        self.statistics = statistics
        self.seed = seed
        for name in config.model.instruments:
            if WAVELENGTH not in axes[name] and name not in statistics:
                raise ValueError(f"{name} has no finite statistics to normalise by")

    def __len__(self) -> int:
        """Return how many features the split holds.

        Returns:
            count: The number of features.
        """
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[dict[str, dict[str, Tensor]], str]:
        """Return one feature's patches of each instrument as tensors, and its class.

        Args:
            index: Which feature of the split.

        Returns:
            sample: For each instrument the model reads, its normalised patches
                under "values" (K, *P), whether each sample of them is a
                measurement under "valid" (K, *P'), and where each sits in
                metres under "position" (K, 3), K being how many the feature
                was drawn with, which may be none.
            feature_class: The class of the feature, as ODE names it.
        """
        rows = rows_by_instrument(self.features[index])
        feature = self.features[index][0].feature
        rng = np.random.default_rng(None if self.seed is None else (self.seed, index))
        held = rows.get(self.config.model.elevation)
        if not held:
            raise ValueError(
                f"{feature} has no {self.config.model.elevation} to stand on"
            )
        elevation = read_elevation(held, self.build)
        sample = {}
        for name in self.config.model.instruments:
            drawn = draw_patches(
                rows.get(name, []),
                self.build,
                self.config.dataset.patchsize,
                self.config.model.patches,
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
                position.append(self.placed(patch, elevation))
            sample[name] = {
                "values": stacked(values, shape),
                "valid": stacked(valid, valid_shape),
                "position": stacked(position, (3,)),
            }
        return sample, feature[0]

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

    def placed(self, patch: Patch, elevation: Elevation) -> np.ndarray:
        """Return where one patch centre sits.

        Args:
            patch: The patch.
            elevation: Where the ground stands over the feature.

        Returns:
            position: East, north and height, in metres. (3,)
        """
        height = patch_elevation(patch, elevation)
        return np.array([patch.east_m, patch.north_m, height], dtype=np.float32)


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
        batch[name] = Tokens(
            values=pad_sequence([one["values"] for one in held], batch_first=True),
            valid=pad_sequence([one["valid"] for one in held], batch_first=True),
            position=pad_sequence([one["position"] for one in held], batch_first=True),
            present=slots.unsqueeze(0) < counts.unsqueeze(1),
        )
    return batch, [feature_class for _, feature_class in samples]
