"""SWIM's targets under one tile beside the head's, a row per target."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from mars_multisensor_dataset.analysis.ground_truth.swim.sample import (
    GRID_POINTS,
    GRID_SPACING_KM,
)
from mars_multisensor_dataset.common.maths.physics import METRES_PER_KM

from dataset.loader import TARGETS
from logs.images.attention import centred_rows, grey_patches, laid_patches, map_places
from logs.images.reconstruction import colour_bar, scaled_values, spaced_row

TITLES = {"attentive-probe": "Attentive Probe", "fine-tune": "Fine-tuning"}

VIRIDIS_RGB = np.array(
    [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)]
)


def coloured(
    grey: np.ndarray, value: np.ndarray, low: float, high: float
) -> np.ndarray:
    """Return a grey image under the colour of its value, as viridis runs low to high.

    Args:
        grey: The image from 0 to 1. (H, W)
        value: What is coloured, clipped to the range. (H, W)
        low: The value coloured at the start of viridis.
        high: The value coloured at its end.

    Returns:
        image: The image. (H, W, 3)
    """
    share = np.clip((value - low) / max(high - low, 1e-6), 0, 1)
    anchors = np.linspace(0, 1, len(VIRIDIS_RGB))
    colour = np.stack(
        [np.interp(share, anchors, channel) for channel in VIRIDIS_RGB.T], -1
    )
    return (0.25 * grey[..., None] * 255 + 0.75 * colour).astype(np.uint8)


def ice_image(
    arrays: Mapping[str, np.ndarray],
    target: np.ndarray,
    predictions: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, str]:
    """Return SWIM's grid beside each protocol's over one instrument, and its caption.

    Args:
        arrays: The observations shown, as the evaluation draws them.
        target: SWIM's value of each point for each target. (P * P, Z)
        predictions: Each point's prediction of each target, keyed by protocol.
            (P * P, Z)

    Returns:
        image: The grids from left to right beside the bar of their colours, a row
            per target in TARGETS order, each coloured over the range its grids span.
            (H, W, 3)
        caption: What the panels are.
    """
    grids = {"Ground Truth": target} | {
        TITLES[protocol]: held for protocol, held in predictions.items()
    }
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
    rows = []
    for at in range(len(TARGETS)):
        shown = np.stack([grid[:, at] for grid in grids.values()])
        low, high = float(shown.min()), float(shown.max())
        ramp = np.linspace(high, low, 256)[:, None]
        legend = colour_bar(
            coloured(np.ones_like(ramp), ramp, low, high)[:, 0],
            [f"{high:.2f}", f"{(low + high) / 2:.2f}", f"{low:.2f}"],
        )
        panels = [coloured(grey, grid[point, at], low, high) for grid in grids.values()]
        rows.append(spaced_row([*panels, legend]))
    return centred_rows(rows), f"{' | '.join(grids)}, rows {', '.join(TARGETS)}"
