"""One observation's patches beside what UMR and CMR rebuilt of them."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from mars_multisensor_dataset.building.configs import sharad
from mars_multisensor_dataset.building.preprocessing.crism.correction import (
    centre_wavelengths,
)
from PIL import Image, ImageDraw, ImageFont

COMPOSITES_NM = {
    "albedo_770nm": (770.0, 770.0, 770.0),
    "false_colour_2529_1506_1080nm": (2529.0, 1506.0, 1080.0),
}
EMPTY_RGB = (255, 255, 255)
SPACE = 12
ROW_HEIGHT = 512


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


def value_range(images: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Return the range images are drawn over, black at its low and white at its high.

    Args:
        images: The images, the original first. (N, ...)
        clip: The low and high the z-scores were clipped to, NaN for an open side,
            which the original's own extreme takes.

    Returns:
        range: The low and high. (2,)
    """
    return np.where(np.isnan(clip), (np.nanmin(images[0]), np.nanmax(images[0])), clip)


def scaled_values(images: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Return images over one range from 0 to 1, NaN kept where nothing is shown.

    Args:
        images: The images, the original first. (N, ...)
        clip: The low and high the z-scores were clipped to, NaN for an open side,
            which the original's own extreme takes.

    Returns:
        scaled: The same images, clipped to that range and scaled to it. (N, ...)
    """
    low, high = value_range(images, clip)
    return np.clip((images - low) / np.maximum(high - low, 1e-12), 0, 1)


def colour_bar(colours: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    """Return a vertical colour bar labelled at its top, middle and bottom.

    Args:
        colours: The bar's colours from top to bottom. (N, 3)
        labels: What its top, middle and bottom stand for.

    Returns:
        bar: The bar and its labels on white, a margin to its left. (ROW_HEIGHT, W, 3)
    """
    size = ROW_HEIGHT // 24
    font = ImageFont.load_default(size=size)
    top, bottom, left = size // 2 + 1, ROW_HEIGHT - size // 2 - 1, size // 2
    text = left + size + size // 2
    width = text + round(max(font.getlength(label) for label in labels)) + left
    image = Image.new("RGB", (width, ROW_HEIGHT), EMPTY_RGB)
    strip = Image.fromarray(np.asarray(colours, np.uint8)[:, None])
    image.paste(
        strip.resize((size, bottom - top), Image.Resampling.NEAREST), (left, top)
    )
    draw = ImageDraw.Draw(image)
    draw.rectangle((left, top, left + size, bottom), outline=(0, 0, 0))
    for label, y in zip(labels, (top, (top + bottom) // 2, bottom), strict=True):
        draw.line((left + size, y, left + size + size // 3, y), fill=(0, 0, 0))
        draw.text((text, y), label, fill=(0, 0, 0), font=font, anchor="lm")
    return np.array(image)


def value_bar(images: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Return the grey bar of images drawn by scaled_values, labelled with their values.

    Args:
        images: The images, the original first. (N, ...)
        clip: The low and high the z-scores were clipped to, NaN for an open side,
            which the original's own extreme takes.

    Returns:
        bar: The bar, white at the top. (ROW_HEIGHT, W, 3)
    """
    low, high = value_range(images, clip)
    return colour_bar(
        np.repeat(np.linspace(255, 0, 256)[:, None], 3, axis=1),
        [f"{value:.3g}" for value in (high, (low + high) / 2, low)],
    )


def sounder_resampled(images: np.ndarray) -> np.ndarray:
    """Return images resampled to ROW_HEIGHT rows and half as many columns.

    Args:
        images: The images. (N, H, W, ...)

    Returns:
        resampled: The same images, each twice as high as wide.
            (N, ROW_HEIGHT, ROW_HEIGHT / 2, ...)
    """
    taken, columns = (
        np.linspace(0, size - 1, wanted).round().astype(int)
        for size, wanted in zip(
            images.shape[1:3], (ROW_HEIGHT, ROW_HEIGHT // 2), strict=True
        )
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
        stretched: Whether each panel is resampled to half as many columns as rows,
            so it stands twice as high as wide.

    Returns:
        image: The panels from left to right and the bar of their values, white a
            twelfth of one around each. (ROW_HEIGHT, W', 3)
    """
    scaled = scaled_values(images, clip) * 255
    rgb = np.repeat(scaled, 3 // images.shape[-1], axis=-1)  # (N, H, W, 3)
    empty = np.isnan(images[..., 0])  # (N, H, W)
    rgb[empty] = EMPTY_RGB
    # The input's hidden patches are the empty ones the original holds
    rgb[1][empty[1] & ~empty[0]] = (150, 175, 215)
    rgb = rgb.astype(np.uint8)
    if stretched:
        panels = list(sounder_resampled(rgb))
    else:
        width = round(rgb.shape[2] * ROW_HEIGHT / rgb.shape[1])
        panels = [
            np.array(
                Image.fromarray(one).resize(
                    (width, ROW_HEIGHT), Image.Resampling.NEAREST
                )
            )
            for one in rgb
        ]
    return spaced_row([*panels, value_bar(images, clip)])


def nearest_bands(wanted: tuple[float, ...]) -> np.ndarray:
    """Return the CRISM band nearest each wavelength.

    Args:
        wanted: The wavelengths, in nanometres.

    Returns:
        bands: The index of each one's band. (len(wanted),)
    """
    centres = np.asarray(centre_wavelengths.LAYOUT.band_centres_nm)
    return np.abs(centres[:, None] - np.asarray(wanted)).argmin(axis=0)


def instrument_panels(
    name: str,
    images: np.ndarray,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return one instrument's panels side by side, a spectrum's as stored composites.

    Args:
        name: The instrument, as ODE names it.
        images: The original, then what is drawn beside it. (N, *image)
        arrays: Its patches as the run read them, with their "clip", "mean" and "std".

    Returns:
        drawn: The panels side by side, keyed by what they show. (ROW_HEIGHT, W, 3)
    """
    if images.ndim == 3:
        return {
            name: panel_image(
                images[..., None],
                arrays["clip"],
                name == sharad.LAYOUT.instrument,
            )
        }
    # A spectrum is drawn as stored, over the original's own range
    stored = images * arrays["std"] + arrays["mean"]
    return {
        f"{name}_{label}": panel_image(
            stored[..., nearest_bands(wanted)], np.full(2, np.nan), False
        )
        for label, wanted in COMPOSITES_NM.items()
    }
