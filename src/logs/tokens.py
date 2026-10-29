"""One tile's shared tokens, their three main directions drawn as colour."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from logs.attention import ROW_HEIGHT, centred_rows, laid_patches, map_frame
from logs.reconstruction import (
    EMPTY_RGB,
    SOUNDER,
    patch_mosaic,
    sounder_resampled,
    spaced_row,
)

COMPONENTS = 3
SPREAD_PERCENT = (1.0, 99.0)


def token_image(
    reconstruction: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[np.ndarray, str]:
    """Return every instrument's tokens coloured by the main directions of them all.

    Args:
        reconstruction: Each instrument's shown observation and its tokens, as the
            evaluation drew them.

    Returns:
        image: Each ground instrument on one map, the sounder's radargram beside.
            (H, W, 3)
        caption: What the colours show.
    """
    # One principal component analysis over every instrument's tokens together, each
    # centred on its own mean so the colours follow what varies, not the instrument
    centred = np.concatenate(
        [held["token"] - held["token"].mean(axis=0) for held in reconstruction.values()]
    )
    _, spread, directions = np.linalg.svd(centred, full_matrices=False)
    projected = centred @ directions[:COMPONENTS].T  # (N, 3)
    low, high = np.percentile(projected, SPREAD_PERCENT, axis=0)
    colours = np.clip((projected - low) / (high - low), 0, 1)
    explained = spread[:COMPONENTS] ** 2 / (spread**2).sum()
    grounds = [held for name, held in reconstruction.items() if name != SOUNDER]
    frame = map_frame(grounds)
    panels, start = [], 0
    for name, held in reconstruction.items():
        mine = colours[start : start + len(held["token"])]  # (K, 3)
        start += len(held["token"])
        if name == SOUNDER:
            shape = held["values"].shape
            # Each patch's colour over its measured samples, along its pass
            rgb = sounder_resampled(
                np.stack(
                    [
                        patch_mosaic(
                            np.broadcast_to(
                                one.reshape(-1, *[1] * (len(shape) - 1)), shape
                            ),
                            held["measured"],
                            held["cell"],
                        )
                        for one in mine.T
                    ],
                    axis=-1,
                )[None],
                ROW_HEIGHT,
            )[0]
        else:
            ground = held["values"].shape[1:3]
            rgb = np.stack(
                [
                    laid_patches(
                        held,
                        np.broadcast_to(one[:, None, None], (len(one), *ground)),
                        frame,
                    )
                    for one in mine.T
                ],
                axis=-1,
            )
        image = np.nan_to_num(rgb) * 255
        image[np.isnan(rgb[..., 0])] = EMPTY_RGB
        panels.append(image.astype(np.uint8))
    caption = (
        f"{', '.join(reconstruction)} tokens in the shared space, every patch read."
        f" Colour: the first {COMPONENTS} principal components of all of them together"
        f" as red, green and blue ({', '.join(f'{one:.0%}' for one in explained)} of"
        " the variance), each instrument centred on its own mean. The same colour"
        " means the model puts those tokens together."
    )
    return centred_rows([spaced_row(panels)]), caption
