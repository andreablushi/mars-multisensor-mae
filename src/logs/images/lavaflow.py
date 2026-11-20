"""A reflector crop beside the crops it retrieves, each framed by its verdict."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from lavaflow.models import (
    NONE,
    REFLECTOR,
)

from logs.images.panels import (
    EMPTY_RGB,
    DrawnImage,
    patch_mosaic,
    scaled_values,
    sounder_resampled,
    spaced_row,
    value_bar,
)

VERDICT_RGB = {REFLECTOR: (40, 160, 60), NONE: (200, 40, 40)}

FRAME = 6


def lavaflow_image(
    drawn: Sequence[Mapping[str, np.ndarray]],
    crops: Sequence[str],
    verdicts: Sequence[str],
    cosines: Sequence[float],
) -> DrawnImage:
    """Return a query's radargram beside those it retrieved, framed green or red.

    Args:
        drawn: The query's SHARAD patches, then each retrieved crop's, as
            drawn_arrays names them.
        crops: The query and each retrieved crop, as "<tile>/<track>".
        verdicts: What the review saw in each retrieved crop.
        cosines: Each retrieved crop's cosine with the query.

    Returns:
        image: The radargrams from left to right over the query's range, and its
            bar. (ROW_HEIGHT, W, 3)
        caption: The query, then each retrieved crop with its cosine and verdict.
    """
    images = np.stack(
        [
            sounder_resampled(
                patch_mosaic(one["values"], one["measured"], one["cell"])[None]
            )[0]
            for one in drawn
        ]
    )  # (N, ROW_HEIGHT, ROW_HEIGHT / 2)
    clip = drawn[0]["clip"]
    rgb = np.repeat(scaled_values(images, clip)[..., None] * 255, 3, axis=-1)
    rgb[np.isnan(images)] = EMPTY_RGB
    rgb = rgb.astype(np.uint8)  # (N, ROW_HEIGHT, ROW_HEIGHT / 2, 3)
    for panel, verdict in zip(rgb[1:], verdicts, strict=True):
        for edge in (
            np.s_[:FRAME],
            np.s_[-FRAME:],
            np.s_[:, :FRAME],
            np.s_[:, -FRAME:],
        ):
            panel[edge] = VERDICT_RGB[verdict]
    caption = f"query {crops[0]} | " + ", ".join(
        f"{crop} {cosine:.3f} {verdict}"
        for crop, cosine, verdict in zip(crops[1:], cosines, verdicts, strict=True)
    )
    return DrawnImage(spaced_row([*rgb, value_bar(images, clip)]), caption)
