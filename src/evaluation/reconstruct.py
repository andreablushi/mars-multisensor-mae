"""Rebuilding one tile's hidden patches, from its own tokens and the others'."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from building.metadata.observation import ObservationMetadata

from architecture.mae import CrossSensorMAE
from architecture.tokens import Tokens, token_batch_padding
from dataset.models.positioning import read_surface_delays
from dataset.patches import cropped_patch_arrays, patch_arrays, read_tile_patches
from dataset.store import DatasetBuild
from training.step import drawn_masks


def reconstruct(
    model: CrossSensorMAE,
    build: DatasetBuild,
    rows: Mapping[str, Sequence[ObservationMetadata]],
    axes: Mapping[str, tuple[str, ...]],
    sizes: Mapping[str, Mapping[str, int]],
    pool: Mapping[str, int],
    shapes: Mapping[str, tuple[int, ...]],
    delay: str,
    budget: Mapping[str, int],
    mask_ratio: float,
    seed: int,
    device: torch.device,
) -> dict[str, dict[str, np.ndarray]]:
    """Return one observation of each instrument held, cut as validation cuts, rebuilt.

    Args:
        model: The model, on the device.
        build: The build the tile is read from.
        rows: The tile's index rows of each instrument.
        axes: What each axis of each instrument's values holds.
        sizes: How far a patch of each sensor runs along each axis it is cut on.
        pool: How many ground samples of a patch each instrument averages into one.
        shapes: The shape of one patch of each instrument as the model reads it.
        delay: The instrument whose rows give every surface patch its delay.
        budget: How many patches of each instrument are kept.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        seed: What fixes the cut and the masks.
        device: Where the model runs.

    Returns:
        reconstruction: Each instrument holding any patch: its patches as the model
            read them under "values" and "measured", their place on the
            observation's grid under "cell", which were hidden under "hidden", and
            the hidden ones rebuilt from its own tokens under "umr" and from the
            others' under "cmr".
    """
    # The largest observation of each the tile holds, so one image of each is drawn
    shown = {
        name: [max(rows[name], key=lambda one: math.prod(one.shape))]
        for name in sizes
        if name in rows
    }
    read = read_tile_patches(
        shown, build, sizes, pool, read_surface_delays(build, rows[delay])
    )
    sample = cropped_patch_arrays(
        {
            name: patch_arrays(held, shapes[name], axes[name])
            | {"cell": np.array([one.cell for one in held])}
            for name, held in read.items()
        },
        budget,
        random.Random(seed),
    )
    batch, _ = token_batch_padding([(sample, "")])
    batch = {
        name: Tokens(*(one.to(device) for one in tokens))
        for name, tokens in batch.items()
    }
    generator = torch.Generator(device=device).manual_seed(seed)
    visible, hidden = drawn_masks(batch, mask_ratio, generator)
    model.eval()
    with torch.no_grad():
        predictions = model(batch, visible, hidden)
    return {
        name: arrays
        | {"hidden": hidden[name][0].cpu().numpy()}
        | {term: predictions[term][name][0].cpu().numpy() for term in predictions}
        for name, arrays in sample.items()
        if len(arrays["values"])
    }
