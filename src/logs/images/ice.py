"""SWIM's ice consistency under one tile beside each head's, a row per depth."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from analysis.ground_truth.swim.sample import GRID_POINTS, GRID_SPACING_KM
from common.maths.physics import METRES_PER_KM

from benchmarking.evaluation.metrics import DEPTHS
from logs.images.attention import (
    centred_rows,
    grey_patches,
    laid_patches,
    map_places,
    signed_overlay,
)
from logs.images.reconstruction import colour_bar, scaled_values, spaced_row


def ice_image(
    arrays: Mapping[str, np.ndarray],
    grids: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, str]:
    """Return every grid over the observations of one instrument, and its caption.

    Args:
        arrays: The observations shown, as the evaluation draws them.
        grids: The consistency of each point at each depth, SWIM's own first,
            keyed by what each shows. (P * P, Z)

    Returns:
        image: The grids from left to right beside the bar of their colours, a row
            per depth in DEPTHS order. (H, W, 3)
        caption: What the panels are.
    """
    half = GRID_POINTS * GRID_SPACING_KM * METRES_PER_KM / 2
    frame = np.array([half, -half, -half, half])
    values, clip = grey_patches(arrays, arrays["values"])
    grey = np.nan_to_num(
        scaled_values(laid_patches(arrays, values, frame)[None], clip)[0], nan=1.0
    )
    places = map_places(frame)  # (H, W, 2)
    row, column = (
        np.clip((offset / (2 * half) * GRID_POINTS).astype(int), 0, GRID_POINTS - 1)
        for offset in (half - places[..., 0], places[..., 1] + half)
    )
    point = row * GRID_POINTS + column  # (H, W)
    lift = np.linspace(1, -1, 256)[:, None]
    bar = signed_overlay(np.ones_like(lift), lift)[:, 0]
    legend = colour_bar(bar, ["1", "0", "-1"])
    rows = [
        spaced_row(
            [signed_overlay(grey, grid[point, at]) for grid in grids.values()]
            + [legend]
        )
        for at in range(len(DEPTHS))
    ]
    return centred_rows(rows), f"{' | '.join(grids)}, rows {', '.join(DEPTHS)}"
