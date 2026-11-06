"""SWIM's ice under one tile beside each head's chance of it, over one observation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from analysis.ground_truth.swim.sample import GRID_POINTS, GRID_SPACING_KM
from common.maths.physics import METRES_PER_KM

from benchmarking.evaluation.metrics import DEPTHS
from logs.images.attention import centred_rows, grey_patches, laid_patches, map_places
from logs.images.reconstruction import colour_bar, scaled_values, spaced_row

ICE_RGB = (215, 48, 39)

DRY_RGB = (69, 117, 180)


def tinted(grey: np.ndarray, chance: np.ndarray) -> np.ndarray:
    """Return a grey image tinted blue where ice is unlikely and red where it is likely.

    Args:
        grey: The image from 0 to 1, NaN where nothing is shown. (H, W)
        chance: The chance of ice there, from 0 to 1. (H, W)

    Returns:
        image: The image, white beneath the tint where nothing is shown. (H, W, 3)
    """
    chance = chance[..., None]
    colour = (1 - chance) * np.array(DRY_RGB) + chance * np.array(ICE_RGB)
    shade = np.nan_to_num(grey, nan=1.0)[..., None] * 255
    return (0.5 * shade + 0.5 * colour).astype(np.uint8)


def ice_image(
    arrays: Mapping[str, np.ndarray], grids: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, str]:
    """Return a row per depth of every grid over one observation, and its caption.

    Args:
        arrays: The observation shown, as the evaluation draws it.
        grids: The chance of ice at every point and depth, SWIM's own first, keyed
            by what each shows. (P * P, Z)

    Returns:
        image: A row per depth from the shallowest, its grids from left to right
            beside the bar of their colours. (H, W, 3)
        caption: What the panels and rows are, and the observation shown.
    """
    half = GRID_POINTS * GRID_SPACING_KM * METRES_PER_KM / 2
    frame = np.array([half, -half, -half, half])
    values, clip = grey_patches(arrays, arrays["values"])
    grey = scaled_values(laid_patches(arrays, values, frame)[None], clip)[0]
    places = map_places(frame)  # (H, W, 2)
    row, column = (
        np.clip((offset / (2 * half) * GRID_POINTS).astype(int), 0, GRID_POINTS - 1)
        for offset in (half - places[..., 0], places[..., 1] + half)
    )
    point = row * GRID_POINTS + column  # (H, W)
    bar = colour_bar(
        tinted(np.ones((256, 1)), np.linspace(1, 0, 256)[:, None])[:, 0],
        ["1", "0.5", "0"],
    )
    rows = [
        spaced_row([*(tinted(grey, grid[point, at]) for grid in grids.values()), bar])
        for at in range(len(DEPTHS))
    ]
    caption = (
        f"{' | '.join(grids)}, rows {' and '.join(DEPTHS)}, "
        f"over {arrays['identifier'][0]}"
    )
    return centred_rows(rows), caption
