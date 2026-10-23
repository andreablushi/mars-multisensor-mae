"""One observation's patches beside what UMR and CMR rebuilt of the hidden ones."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from building.preprocessing.crism.correction.centre_wavelengths import LAYOUT

PANELS = ("original", "input", "UMR", "CMR")
COMPOSITES_NM = {
    "albedo_770nm": (770.0, 770.0, 770.0),
    "false_colour_2529_1506_1080nm": (2529.0, 1506.0, 1080.0),
}
SOUNDER = "SHARAD"
EMPTY_RGB = (255, 255, 255)
HIDDEN_RGB = (150, 175, 215)
SOUNDER_ROWS = 1024
SPACE = 12


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


def panel_image(images: np.ndarray, clip: np.ndarray, stretched: bool) -> np.ndarray:
    """Return the four panels of one image side by side, every band over one range.

    Args:
        images: The original, the input, the UMR and the CMR image, NaN where
            nothing is shown, of one channel or three. (4, H, W, C)
        clip: The low and high the z-scores were clipped to, NaN for an open side,
            which the original's own extreme takes.
        stretched: Whether each panel is resampled to SOUNDER_ROWS rows and half as
            many columns, so it stands twice as high as wide.

    Returns:
        image: The panels from left to right, white a twelfth of one around each.
            (H', W', 3)
    """
    low, high = np.where(
        np.isnan(clip), (np.nanmin(images[0]), np.nanmax(images[0])), clip
    )
    scaled = np.clip((images - low) / np.maximum(high - low, 1e-12), 0, 1) * 255
    rgb = np.repeat(scaled, 3 // images.shape[-1], axis=-1)  # (4, H, W, 3)
    empty = np.isnan(images[..., 0])  # (4, H, W)
    rgb[empty] = EMPTY_RGB
    # The input's hidden patches are the empty ones the original holds
    rgb[1][empty[1] & ~empty[0]] = HIDDEN_RGB
    if stretched:
        rows, columns = (
            np.linspace(0, size - 1, wanted).round().astype(int)
            for size, wanted in zip(
                rgb.shape[1:3], (SOUNDER_ROWS, SOUNDER_ROWS // 2), strict=True
            )
        )
        rgb = rgb[:, rows][:, :, columns]
    space = np.full((rgb.shape[1], max(rgb.shape[2] // SPACE, 1), 3), EMPTY_RGB)
    return np.concatenate(
        [part for panel in rgb for part in (space, panel)] + [space], axis=1
    ).astype(np.uint8)


def reconstruction_images(
    reconstruction: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Return every instrument's image as the run read it, CRISM's as stored composites.

    Args:
        reconstruction: Each instrument's patches, as the run read and rebuilt them.

    Returns:
        drawn: Each image's panels side by side, keyed by what it shows. (H, W, 3)
    """
    centres = np.asarray(LAYOUT.band_centres_nm)
    drawn = {}
    for name, arrays in reconstruction.items():
        values = arrays["values"]
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
            drawn[name] = panel_image(
                images[..., None], arrays["clip"], name == SOUNDER
            )
            continue
        # A spectrum is drawn as stored, over the original's own range
        stored = images * arrays["std"] + arrays["mean"]
        # Each band the nearest to its wavelength
        for label, wanted in COMPOSITES_NM.items():
            bands = np.abs(centres[:, None] - np.asarray(wanted)).argmin(axis=0)
            drawn[f"{name}_{label}"] = panel_image(
                stored[..., bands], np.full(2, np.nan), False
            )
    return drawn
