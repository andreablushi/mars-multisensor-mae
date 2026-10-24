"""Rebuilding one observation's hidden patches, from its own tokens and the others'."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from building.common.layout import Axis
from building.metadata.observation import ObservationMetadata

from architecture.mae import CrossSensorMAE
from architecture.tokens import Tokens, token_batch_padding
from dataset.patches import patch_arrays, read_tile_patches
from dataset.store import DatasetBuild
from evaluation.similarity import GAP
from training.step import drawn_masks, mask_generators

SOUNDER_INCIDENCE_DEG = 140.0


def shown_observation(
    rows: Sequence[ObservationMetadata], axes: Sequence[str]
) -> ObservationMetadata:
    """Return the observation one instrument of a tile is drawn by.

    Args:
        rows: The tile's index rows of the instrument.
        axes: What each axis of its values holds.

    Returns:
        shown: The most measured observation, a sounder's the pass nearest
            SOUNDER_INCIDENCE_DEG.
    """
    if Axis.DELAY in axes:
        return min(
            rows,
            key=lambda one: abs(one.acquisition.incidence_deg - SOUNDER_INCIDENCE_DEG),
        )
    return max(rows, key=lambda one: one.valid_count)


def drawn_stats(build: DatasetBuild, name: str) -> dict[str, np.ndarray]:
    """Return what one instrument's z-scores are drawn by.

    Args:
        build: The build the instrument is read from.
        name: The instrument, as ODE names it.

    Returns:
        stats: The "clip" they were cut to, and the "mean" and "std" that undo them.
    """
    return {"clip": np.array(build.clips[name], dtype=float)} | {
        key: np.array(build.read_stats()[name][key], dtype=float)
        for key in ("mean", "std")
    }


def tile_batch(
    sample: Mapping[str, Mapping[str, np.ndarray]], device: torch.device
) -> dict[str, Tokens]:
    """Return one tile's patch arrays as a batch of one, on the device.

    Args:
        sample: Each instrument's patch arrays.
        device: Where the model runs.

    Returns:
        batch: Each instrument's patches, as the model reads them.
    """
    batch, _ = token_batch_padding([(sample, "")])
    return {
        name: Tokens(*(one.to(device) for one in tokens))
        for name, tokens in batch.items()
    }


def reconstruct(
    model: CrossSensorMAE,
    build: DatasetBuild,
    rows: Mapping[str, Sequence[ObservationMetadata]],
    axes: Mapping[str, tuple[str, ...]],
    sizes: Mapping[str, Mapping[str, int]],
    shapes: Mapping[str, tuple[int, ...]],
    mask_ratio: float,
    seed: int,
    device: torch.device,
) -> dict[str, dict[str, np.ndarray]]:
    """Return one whole observation of each instrument the tile holds, rebuilt.

    Args:
        model: The model, on the device.
        build: The build the tile is read from.
        rows: The tile's index rows of each instrument.
        axes: What each axis of each instrument's values holds.
        sizes: How far a patch of each sensor runs along each axis it is cut on.
        shapes: The shape of one patch of each instrument as the model reads it.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        seed: What fixes the masks.
        device: Where the model runs.

    Returns:
        reconstruction: Each instrument's patches under "values", "measured" and
            "cell", the "clip", "mean" and "std" of their z-scores, the "hidden" ones,
            and those rebuilt under "umr" and "cmr".
    """
    shown = {name: [shown_observation(rows[name], axes[name])] for name in sizes}
    read = read_tile_patches(shown, build, sizes)
    sample = {
        name: patch_arrays(
            held, shapes[name], axes[name], build.condition_width(name, shapes[name])
        )
        | {"cell": np.array([one.cell for one in held])}
        for name, held in read.items()
    }
    batch = tile_batch(sample, device)
    visible, hidden = drawn_masks(
        batch, mask_ratio, mask_generators(batch, seed, device)
    )
    model.eval()
    with torch.no_grad():
        predictions, _ = model(batch, visible, hidden, GAP)
    return {
        name: arrays
        | drawn_stats(build, name)
        | {"hidden": hidden[name][0].cpu().numpy()}
        | {term: predictions[term][name][0].cpu().numpy() for term in predictions}
        for name, arrays in sample.items()
    }
