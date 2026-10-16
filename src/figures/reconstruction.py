"""One tile's patches beside what UMR and CMR rebuilt of the hidden ones."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from building.preprocessing.crism.correction.centre_wavelengths import LAYOUT

from figures.style import TEMPLATE

PANELS = ("original", "input", "UMR", "CMR")
COMPOSITES_NM = {
    "visible": (597.0, 532.0, 441.0),
    "infrared false colour": (2532.0, 1503.0, 1083.0),
}
SOUNDER = "SHARAD"
EMPTY_RGB = (255, 255, 255)
HIDDEN_RGB = (246, 196, 160)
PANEL = 300
SOUNDER_HEIGHT = 700


def patch_mosaic(patches: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Return patches laid side by side where they sit on their observation.

    Args:
        patches: The patches. (K, *P)
        cells: Where each sits on the observation's grid of patches. (K, A)

    Returns:
        mosaic: Every patch at its place, NaN where none is. (*A * P)
    """
    shape = np.array(patches.shape[1:])
    origin = cells - cells.min(axis=0)  # (K, A)
    mosaic = np.full(tuple((origin.max(axis=0) + 1) * shape), np.nan, np.float32)
    for patch, at in zip(patches, origin, strict=True):
        mosaic[
            tuple(
                slice(start, start + size)
                for start, size in zip(at * shape, shape, strict=True)
            )
        ] = patch
    return mosaic


def image_figure(images: np.ndarray, title: str, stretched: bool) -> go.Figure:
    """Draw the four panels of one image, each channel stretched as the original's.

    Args:
        images: The original, the input, the UMR and the CMR image, NaN where
            nothing is shown, of one channel or three. (4, H, W, C)
        title: What the image shows.
        stretched: Whether its pixels are widened so the panels stand twice as
            high as wide.

    Returns:
        figure: The four panels side by side.
    """
    low, high = np.nanpercentile(images[0], (2, 98), axis=(0, 1))  # (C,)
    scaled = np.clip((images - low) / np.maximum(high - low, 1e-6), 0, 1) * 255
    rgb = np.repeat(scaled, 3 // images.shape[-1], axis=-1)  # (4, H, W, 3)
    empty = np.isnan(images[..., 0])  # (4, H, W)
    rgb[empty] = EMPTY_RGB
    # The input's hidden patches are the empty ones the original holds
    rgb[1][empty[1] & ~empty[0]] = HIDDEN_RGB
    figure = px.imshow(
        rgb.astype(np.uint8),
        facet_col=0,
        facet_col_spacing=0.01,
        title=title,
    )
    for annotation, panel in zip(figure.layout.annotations, PANELS, strict=True):
        annotation.text = panel
    figure.update_xaxes(showticklabels=False)
    figure.update_yaxes(showticklabels=False)
    height, width = images.shape[1:3]
    figure.update_traces(
        hoverinfo="skip",
        hovertemplate=None,
        dx=height / (2 * width) if stretched else 1,
    )
    figure.update_layout(
        template=TEMPLATE,
        width=4 * PANEL,
        height=SOUNDER_HEIGHT if stretched else PANEL + 80,
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
    )
    return figure


def reconstruction_figures(
    reconstruction: Mapping[str, Mapping[str, np.ndarray]],
) -> list[go.Figure]:
    """Draw every instrument's image, CRISM's as two colour composites."""
    centres = np.asarray(LAYOUT.band_centres_nm)
    figures = []
    for name, arrays in reconstruction.items():
        values = arrays["values"]
        normalization = f"normalised by {arrays['normalization']}"
        hidden = arrays["hidden"].reshape(-1, *[1] * (values.ndim - 1))
        images = np.stack(
            [
                patch_mosaic(np.where(arrays["measured"], one, np.nan), arrays["cell"])
                for one in (
                    values,
                    np.where(hidden, np.nan, values),
                    np.where(hidden, arrays["umr"], values),
                    np.where(hidden, arrays["cmr"], values),
                )
            ]
        )  # (4, *image)
        if images.ndim == 3:
            figures.append(
                image_figure(
                    images[..., None], f"{name}, {normalization}", name == SOUNDER
                )
            )
            continue
        for label, wanted in COMPOSITES_NM.items():
            bands = np.abs(centres[:, None] - np.asarray(wanted)).argmin(axis=0)
            nm = ", ".join(f"{centres[at]:.0f}" for at in bands)
            figures.append(
                image_figure(
                    images[..., bands],
                    f"{name} {label} ({nm} nm), {normalization}",
                    False,
                )
            )
    return figures
