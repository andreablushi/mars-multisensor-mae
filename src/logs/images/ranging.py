"""One ranging instrument rebuilt from the other alone, and where its CMR looked."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from building.configs import sharad

from logs.images.attention import (
    TRACE_RGB,
    attention_image,
    centred_rows,
    grey_patches,
    laid_patches,
    map_frame,
    observation_trace,
    overlaid,
    ratio_bar,
    sounder_passes,
)
from logs.images.reconstruction import scaled_values, spaced_row, value_bar


def ranging_image(
    attention: Mapping[str, Mapping[str, np.ndarray]], query: str
) -> tuple[np.ndarray, str]:
    """Return one ranging instrument's original and CMR from the other alone.

    Args:
        attention: The query's and the other's patches, as the evaluation drew them.
        query: The ranging instrument rebuilt.

    Returns:
        image: The sounder drawn as attention_image draws it, a ground query's two
            panels with the most attended tracks above those passes' radargrams,
            each row centred. (H, W, 3)
        caption: What is rebuilt from what, and each pass's share per token.
    """
    if query == sharad.LAYOUT.instrument:
        return attention_image(attention, query)
    arrays, held = attention[query], attention[sharad.LAYOUT.instrument]
    frame = map_frame([arrays])
    laid = []
    for one in (arrays["values"], arrays["cmr"]):
        grey, clip = grey_patches(arrays, one)
        laid.append(laid_patches(arrays, grey, frame))
    passes = sounder_passes(held, 3)
    track = np.any(
        [
            observation_trace(
                {key: held[key][mine] for key in ("cell", "position")}, frame, True
            )
            for mine, *_ in passes
        ],
        axis=0,
    )
    top = []
    for grey in scaled_values(np.stack(laid), clip):
        panel = overlaid(grey, np.ones_like(grey))
        panel[track & ~np.isnan(grey)] = TRACE_RGB
        top.append(panel)
    top.append(value_bar(np.stack(laid), clip))
    radargrams = [
        np.concatenate([overlaid(grey, ratio / held["attention"].mean()), bar], axis=1)
        for _, grey, bar, ratio, *_ in passes
    ]
    radargrams.append(ratio_bar())
    caption = f"{query} from {sharad.LAYOUT.instrument}: " + ", ".join(
        f"pass {identifier} {share:.0%}" for *_, identifier, share in passes
    )
    return centred_rows([spaced_row(top), spaced_row(radargrams)]), caption
