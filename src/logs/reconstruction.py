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


def patch_mosaic(
    patches: np.ndarray, measured: np.ndarray, cells: np.ndarray
) -> np.ndarray:
    """Return patches laid side by side where they sit on their observation.

    Args:
        patches: The patches. (K, *P)
        measured: Which of their samples are measurements. (K, *P')
        cells: Where each sits on the observation's grid of patches. (K, A)

    Returns:
        mosaic: Every measured sample at its place, NaN where none is. (*A * P)
    """
    patches = np.where(measured, patches, np.nan)
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


def scaled_values(images: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Return images over one range from 0 to 1, NaN kept where nothing is shown.

    Args:
        images: The images, the original first. (N, ...)
        clip: The low and high the z-scores were clipped to, NaN for an open side,
            which the original's own extreme takes.

    Returns:
        scaled: The same images, clipped to that range and scaled to it. (N, ...)
    """
    low, high = np.where(
        np.isnan(clip), (np.nanmin(images[0]), np.nanmax(images[0])), clip
    )
    return np.clip((images - low) / np.maximum(high - low, 1e-12), 0, 1)


def sounder_resampled(images: np.ndarray, rows: int) -> np.ndarray:
    """Return images resampled to some rows and half as many columns.

    Args:
        images: The images. (N, H, W, ...)
        rows: How many rows each is resampled to.

    Returns:
        resampled: The same images, each twice as high as wide. (N, rows, rows / 2, ...)
    """
    taken, columns = (
        np.linspace(0, size - 1, wanted).round().astype(int)
        for size, wanted in zip(images.shape[1:3], (rows, rows // 2), strict=True)
    )
    return images[:, taken][:, :, columns]


def spaced_row(panels: list[np.ndarray]) -> np.ndarray:
    """Return panels of one height side by side, white a twelfth of one around each.

    Args:
        panels: The panels. (H, W, 3) each

    Returns:
        row: The panels from left to right. (H, W', 3)
    """
    space = np.full(
        (panels[0].shape[0], max(panels[0].shape[1] // SPACE, 1), 3), EMPTY_RGB
    )
    return np.concatenate(
        [part for panel in panels for part in (space, panel)] + [space], axis=1
    ).astype(np.uint8)


def panel_image(images: np.ndarray, clip: np.ndarray, stretched: bool) -> np.ndarray:
    """Return the panels of one image side by side, every band over one range.

    Args:
        images: The original, then what is drawn beside it, the second's hidden
            patches the empty ones the original holds, NaN where nothing is shown, of
            one channel or three. (N, H, W, C)
        clip: The low and high the z-scores were clipped to, NaN for an open side,
            which the original's own extreme takes.
        stretched: Whether each panel is resampled to SOUNDER_ROWS rows and half as
            many columns, so it stands twice as high as wide.

    Returns:
        image: The panels from left to right, white a twelfth of one around each.
            (H', W', 3)
    """
    scaled = scaled_values(images, clip) * 255
    rgb = np.repeat(scaled, 3 // images.shape[-1], axis=-1)  # (N, H, W, 3)
    empty = np.isnan(images[..., 0])  # (N, H, W)
    rgb[empty] = EMPTY_RGB
    # The input's hidden patches are the empty ones the original holds
    rgb[1][empty[1] & ~empty[0]] = HIDDEN_RGB
    if stretched:
        rgb = sounder_resampled(rgb, SOUNDER_ROWS)
    return spaced_row(list(rgb))


def nearest_bands(wanted: tuple[float, ...]) -> np.ndarray:
    """Return the CRISM band nearest each wavelength.

    Args:
        wanted: The wavelengths, in nanometres.

    Returns:
        bands: The index of each one's band. (len(wanted),)
    """
    centres = np.asarray(LAYOUT.band_centres_nm)
    return np.abs(centres[:, None] - np.asarray(wanted)).argmin(axis=0)


def instrument_panels(
    name: str,
    images: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    composites: Mapping[str, tuple[float, ...]],
) -> dict[str, np.ndarray]:
    """Return one instrument's panels side by side, a spectrum's as stored composites.

    Args:
        name: The instrument, as ODE names it.
        images: The original, then what is drawn beside it. (N, *image)
        arrays: Its patches as the run read them, with their "clip", "mean" and "std".
        composites: The wavelengths of each composite a spectrum is drawn as.

    Returns:
        drawn: The panels side by side, keyed by what they show. (H, W, 3)
    """
    if images.ndim == 3:
        return {name: panel_image(images[..., None], arrays["clip"], name == SOUNDER)}
    # A spectrum is drawn as stored, over the original's own range
    stored = images * arrays["std"] + arrays["mean"]
    return {
        f"{name}_{label}": panel_image(
            stored[..., nearest_bands(wanted)], np.full(2, np.nan), False
        )
        for label, wanted in composites.items()
    }


def reconstruction_images(
    reconstruction: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Return every instrument's image as the run read it, CRISM's as stored composites.

    Args:
        reconstruction: Each instrument's patches, as the run read and rebuilt them.

    Returns:
        drawn: Each image's panels side by side, keyed by what it shows. (H, W, 3)
    """
    drawn = {}
    for name, arrays in reconstruction.items():
        values = arrays["values"]
        hidden = arrays["hidden"].reshape(-1, *[1] * (values.ndim - 1))
        images = np.stack(
            [
                patch_mosaic(one, arrays["measured"], arrays["cell"])
                for one in (
                    values,
                    np.where(hidden, np.nan, values),
                    np.where(hidden, arrays["umr"], values),
                    np.where(hidden, arrays["cmr"], values),
                )
            ]
        )  # (4, *image)
        drawn |= instrument_panels(name, images, arrays, COMPOSITES_NM)
    return drawn
