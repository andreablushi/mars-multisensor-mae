"""One observation's patches beside what UMR and CMR rebuilt of the hidden ones."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from building.configs.crism import Detector
from building.preprocessing.crism.correction.centre_wavelengths import LAYOUT
from visualization.training_qualitative.config import VIEWS

from figures.style import TEMPLATE

PANELS = ("original", "input", "UMR", "CMR")
COMPOSITES_NM = {
    "VNIR albedo 770 nm": (770.0, 770.0, 770.0),
    **{
        label: bands
        for label, (detector, bands) in VIEWS.items()
        if detector == Detector.INFRARED
    },
}
SOUNDER = "SHARAD"
EMPTY_RGB = (255, 255, 255)
HIDDEN_RGB = (150, 175, 215)
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


def image_figure(
    images: np.ndarray, title: str, note: str, stretched: bool
) -> go.Figure:
    """Draw the four panels of one image, every band over one 2-98% of the original.

    Args:
        images: The original, the input, the UMR and the CMR image, NaN where
            nothing is shown, of one channel or three. (4, H, W, C)
        title: What the image shows.
        note: How its values were read, written under the title with the range.
        stretched: Whether its pixels are widened so the panels stand twice as
            high as wide.

    Returns:
        figure: The four panels side by side.
    """
    # One range for every band, so the bands keep the balance the model reads
    low, high = np.nanpercentile(images[0], (2, 98))
    scaled = np.clip((images - low) / np.maximum(high - low, 1e-12), 0, 1) * 255
    rgb = np.repeat(scaled, 3 // images.shape[-1], axis=-1)  # (4, H, W, 3)
    empty = np.isnan(images[..., 0])  # (4, H, W)
    rgb[empty] = EMPTY_RGB
    # The input's hidden patches are the empty ones the original holds
    rgb[1][empty[1] & ~empty[0]] = HIDDEN_RGB
    figure = px.imshow(
        rgb.astype(np.uint8),
        facet_col=0,
        facet_col_spacing=0.01,
        title=f"{title}<br><sup>{note}, every band over the original's 2-98%</sup>",
    )
    for annotation, panel in zip(figure.layout.annotations, PANELS, strict=True):
        annotation.text = panel
    # A black frame around each panel's whole grid, blank cells included
    frame = {"showticklabels": False, "showline": True, "linecolor": "black"}
    figure.update_xaxes(frame, mirror=True)
    figure.update_yaxes(frame, mirror=True)
    height, width = images.shape[1:3]
    figure.update_traces(
        hoverinfo="skip",
        hovertemplate=None,
        dx=height / (2 * width) if stretched else 1,
    )
    figure.update_layout(
        template=TEMPLATE,
        width=4 * PANEL,
        height=SOUNDER_HEIGHT if stretched else PANEL + 110,
        margin={"l": 10, "r": 10, "t": 100, "b": 10},
    )
    return figure


def reconstruction_figures(
    reconstruction: Mapping[str, Mapping[str, np.ndarray]], run: str
) -> dict[str, go.Figure]:
    """Draw every instrument's image as the run read it, CRISM's as two composites.

    Args:
        reconstruction: Each instrument's patches, as the run read and rebuilt them.
        run: The run, named in every title.

    Returns:
        figures: Each image's figure, keyed by what it shows.
    """
    centres = np.asarray(LAYOUT.band_centres_nm)
    figures = {}
    for name, arrays in reconstruction.items():
        values = arrays["values"]
        note = f"normalised by {arrays['normalization']}"
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
            figures[name] = image_figure(
                images[..., None], f"{name} | {run}", note, name == SOUNDER
            )
            continue
        # Each band the nearest to its wavelength, the infrared as the dataset draws it
        for label, wanted in COMPOSITES_NM.items():
            bands = np.abs(centres[:, None] - np.asarray(wanted)).argmin(axis=0)
            figures[f"{name} {label}"] = image_figure(
                images[..., bands], f"{name} {label} | {run}", note, False
            )
    return figures
