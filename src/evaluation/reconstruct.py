"""Rebuilding one observation's hidden patches, from its own tokens and the others'."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from building.metadata.observation import ObservationMetadata

from architecture.mae import CrossSensorMAE
from architecture.tokens import Tokens, token_batch_padding
from dataset.patches import patch_arrays, read_tile_patches
from dataset.store import DatasetBuild
from evaluation.similarity import GAP
from training.step import drawn_masks, mask_generators


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
            "cell", their "normalization", the "hidden" ones, and those rebuilt
            under "umr" and "cmr".
    """
    # The most measured observation of each the tile holds, read whole
    shown = {name: [max(rows[name], key=lambda one: one.valid_count)] for name in sizes}
    read = read_tile_patches(shown, build, sizes)
    sample = {
        name: patch_arrays(
            held, shapes[name], axes[name], build.condition_width(name, shapes[name])
        )
        | {"cell": np.array([one.cell for one in held])}
        for name, held in read.items()
    }
    batch, _ = token_batch_padding([(sample, "")])
    batch = {
        name: Tokens(*(one.to(device) for one in tokens))
        for name, tokens in batch.items()
    }
    visible, hidden = drawn_masks(
        batch, mask_ratio, mask_generators(batch, seed, device)
    )
    model.eval()
    with torch.no_grad():
        predictions, _ = model(batch, visible, hidden, GAP)
    return {
        name: arrays
        | {"normalization": np.array(f"zscore, clipped to {build.clips[name]}")}
        | {"hidden": hidden[name][0].cpu().numpy()}
        | {term: predictions[term][name][0].cpu().numpy() for term in predictions}
        for name, arrays in sample.items()
    }
