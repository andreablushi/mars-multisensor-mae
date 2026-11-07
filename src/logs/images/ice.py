"""SWIM's ice under one tile beside each head's, depths combined, over a mosaic."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from analysis.ground_truth.swim.sample import GRID_POINTS, GRID_SPACING_KM
from common.maths.physics import METRES_PER_KM

from benchmarking.evaluation.metrics import COMBOS, ice_combos
from logs.images.attention import centred_rows, grey_patches, laid_patches, map_places
from logs.images.reconstruction import colour_bar, scaled_values, spaced_row

COMBO_RGB = {
    "1-5 m only": (253, 174, 97),
    "below 5 m only": (117, 112, 179),
    "both": (215, 48, 39),
}


def tinted(grey: np.ndarray, combos: np.ndarray) -> np.ndarray:
    """Return a grey image tinted by where ice is found, left grey where it is not.

    Args:
        grey: The image from 0 to 1, NaN where nothing is shown. (H, W)
        combos: The index in COMBOS of the depths icy there. (H, W)

    Returns:
        image: The image, white beneath the tint where nothing is shown. (H, W, 3)
    """
    shade = np.repeat(np.nan_to_num(grey, nan=1.0)[..., None] * 255, 3, axis=-1)
    for name, colour in COMBO_RGB.items():
        held = combos == COMBOS.index(name)
        shade[held] = 0.5 * shade[held] + 0.5 * np.array(colour)
    return shade.astype(np.uint8)


def ice_image(
    arrays: Mapping[str, np.ndarray],
    grids: Mapping[str, np.ndarray],
    threshold: float,
) -> tuple[np.ndarray, str]:
    """Return every grid over the observations of one instrument, and its caption.

    Args:
        arrays: The observations shown, as the evaluation draws them.
        grids: The chance of ice at every point and depth, SWIM's own first, keyed
            by what each shows. (P * P, Z)
        threshold: The chance at or above which a depth counts as icy.

    Returns:
        image: The grids from left to right beside the legend of their colours.
            (H, W, 3)
        caption: What the panels are.
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
    shown = [*COMBO_RGB][::-1]
    legend = colour_bar(
        tinted(
            np.ones((len(shown), 1)), np.array([[COMBOS.index(one)] for one in shown])
        )[:, 0],
        shown,
    )
    panels = [
        tinted(grey, ice_combos(grid, threshold)[point]) for grid in grids.values()
    ]
    return centred_rows([spaced_row([*panels, legend])]), " | ".join(grids)
