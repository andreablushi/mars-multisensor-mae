"""One instrument of one tile laid on its patch grid: read, shown and written back."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from building.preprocessing.crism.correction.centre_wavelengths import BANDS_NM
from plotly.subplots import make_subplots

from evaluation.results import Reconstruction
from visualization.style import MISSING_COLOR, TEMPLATE

COMPOSITES = {
    "TRU": (600.0, 530.0, 440.0),
    "FAL": (2529.0, 1506.0, 1080.0),
}
STRETCH = (2, 98)


def patch_canvas(patches: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Return the patches laid on their grid, NaN where none lands.

    Args:
        patches: The patches, their two grid axes first. (K, H, W, ...)
        cells: The row and column of each. (K, 2)

    Returns:
        canvas: The grid, rows then columns of samples. (R * H, C * W, ...)
    """
    rows, columns = (cells - cells.min(0)).T
    height, width = patches.shape[1:3]
    canvas = np.full(
        ((rows.max() + 1) * height, (columns.max() + 1) * width, *patches.shape[3:]),
        np.nan,
    )
    for row, column, patch in zip(rows, columns, patches, strict=True):
        canvas[
            row * height : (row + 1) * height, column * width : (column + 1) * width
        ] = patch
    return canvas


def reconstruction_panels(one: Reconstruction) -> dict[str, np.ndarray]:
    """Return the original, the input and both write-backs, each on the patch grid.

    Args:
        one: One instrument's reconstruction of one tile.

    Returns:
        panels: Each view's canvas, the original left out where nothing hidden
            was measured.
    """
    measured = np.broadcast_to(one.measured, one.values.shape)
    original = np.where(measured, one.values, np.nan)
    shown = one.visible.reshape(-1, *[1] * (one.values.ndim - 1))
    written = one.hidden.reshape(shown.shape)
    panels = {
        "original": original,
        "input": np.where(shown, original, np.nan),
        "UMR": np.where(written, one.umr, np.where(shown, original, np.nan)),
        "CMR": np.where(written, one.cmr, np.where(shown, original, np.nan)),
    }
    if not measured[one.hidden].any():
        del panels["original"]
    return {name: patch_canvas(held, one.cell) for name, held in panels.items()}


def stretched_composite(
    canvas: np.ndarray, reference: np.ndarray, wavelengths: tuple[float, ...]
) -> np.ndarray:
    """Return three bands as an RGB image, stretched as the reference is.

    Args:
        canvas: The spectral canvas to colour. (Y, X, B)
        reference: The canvas the stretch is read from. (Y, X, B)
        wavelengths: The red, green and blue wavelengths, in nm.

    Returns:
        image: The composite, 0 to 255, MISSING_COLOR where nothing is. (Y, X, 3)
    """
    bands = [int(np.argmin(np.abs(np.asarray(BANDS_NM) - one))) for one in wavelengths]
    low, high = np.nanpercentile(reference[..., bands], STRETCH, axis=(0, 1))
    scaled = (canvas[..., bands] - low) / np.maximum(high - low, 1e-6)
    image = np.clip(scaled, 0, 1) * 255
    return np.where(np.isnan(image), MISSING_COLOR, image).astype(np.uint8)


def reconstruction_figure(one: Reconstruction, title: str) -> go.Figure:
    """Draw one instrument of one tile read, shown and written back side by side."""
    panels = reconstruction_panels(one)
    reference = next(iter(panels.values()))
    spectral = reference.ndim == 3
    rows = list(COMPOSITES) if spectral else [None]
    figure = make_subplots(
        rows=len(rows),
        cols=len(panels),
        subplot_titles=[
            name if row is None else f"{name} {row}" for row in rows for name in panels
        ],
        horizontal_spacing=0.02,
        vertical_spacing=0.06,
    )
    low, high = None, None
    if not spectral:
        low, high = np.nanpercentile(reference, STRETCH)
    for at, row in enumerate(rows, start=1):
        for column, canvas in enumerate(panels.values(), start=1):
            trace = (
                go.Image(
                    z=stretched_composite(canvas, reference, COMPOSITES[row]),
                    hoverinfo="skip",
                )
                if spectral
                else go.Heatmap(
                    z=canvas,
                    colorscale="gray",
                    zmin=low,
                    zmax=high,
                    showscale=False,
                    hoverinfo="skip",
                )
            )
            figure.add_trace(trace, row=at, col=column)
    figure.update_xaxes(showticklabels=False)
    figure.update_yaxes(showticklabels=False, autorange="reversed")
    height, width = reference.shape[:2]
    side = 320
    figure.update_layout(
        template=TEMPLATE,
        plot_bgcolor="rgb{}".format(MISSING_COLOR),
        title=title,
        height=120 + len(rows) * side * float(np.clip(height / width, 0.5, 2.5)),
        width=60 + len(panels) * side,
    )
    return figure
