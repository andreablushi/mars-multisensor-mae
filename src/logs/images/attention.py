"""One instrument rebuilt from others whole, and where its CMR looked on each."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from mars_multisensor_dataset.building.configs import sharad

from logs.images.maps import (
    grey_patches,
    ground_samples,
    map_frame,
    map_places,
    observation_trace,
)
from logs.images.panels import (
    EMPTY_RGB,
    DrawnImage,
    centred_rows,
    colour_bar,
    instrument_panels,
    patch_mosaic,
    scaled_values,
    sounder_resampled,
    spaced_row,
    value_bar,
)

TRACE_RGB = (255, 140, 0)


def ground_map(
    arrays: Mapping[str, np.ndarray], frame: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one ground instrument's observations laid on a map, and its attention.

    Args:
        arrays: Its patches, as the evaluation drew them.
        frame: The map's north, south, west and east edges, in metres.

    Returns:
        grey: Its values from 0 to 1, NaN where none is measured, a spectrum's
            770 nm albedo band as stored. (ROW_HEIGHT, W)
        bar: The bar of those values. (ROW_HEIGHT, W', 3)
        ratio: The attention the tokens there drew on average, relative to uniform,
            NaN where no token drew any. (ROW_HEIGHT, W)
    """
    values, clip = grey_patches(arrays, arrays["values"])
    places = map_places(frame)
    grey = np.full(places.shape[:2], np.nan)
    total, count = np.zeros((2, *grey.shape))
    for shown, patch, inner in ground_samples(arrays, places):
        grey[shown] = values[patch, *inner.T]
        drew = arrays["attention"][patch]
        total[shown] += np.nan_to_num(drew)
        count[shown] += ~np.isnan(drew)
    return (
        scaled_values(grey[None], clip)[0],
        value_bar(grey[None], clip),
        np.where(count > 0, total / np.maximum(count, 1), np.nan),
    )


def sounder_passes(
    arrays: Mapping[str, np.ndarray], count: int
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, float]]:
    """Return the sounder's passes whose tokens drew the most attention each.

    Args:
        arrays: Its patches, as the evaluation drew them.
        count: How many passes are returned, the most attended first.

    Returns:
        passes: Each pass's patches (K,), radargram 0 to 1 or NaN and attention over
            uniform (ROW_HEIGHT, ROW_HEIGHT / 2), bar, identifier and per-token share.
    """
    by_pass = np.bincount(arrays["observation"], weights=arrays["overall"])
    by_pass /= np.bincount(arrays["observation"])
    passes = []
    for chosen in by_pass.argsort()[::-1][:count]:
        mine = arrays["observation"] == chosen
        values = arrays["values"][mine]
        attention = np.broadcast_to(
            arrays["attention"][mine].reshape(-1, *[1] * (values.ndim - 1)),
            values.shape,
        )
        grey, ratio = sounder_resampled(
            np.stack(
                [
                    patch_mosaic(one, arrays["measured"][mine], arrays["cell"][mine])
                    for one in (values, attention)
                ]
            ),
        )
        passes.append(
            (
                mine,
                scaled_values(grey[None], arrays["clip"])[0],
                value_bar(grey[None], arrays["clip"]),
                ratio,
                str(arrays["identifier"][chosen]),
                by_pass[chosen] / by_pass.sum(),
            )
        )
    return passes


def overlaid(grey: np.ndarray, ratio: np.ndarray) -> np.ndarray:
    """Return a grey image coloured red where attention is above uniform, blue below.

    Args:
        grey: The image from 0 to 1, NaN where nothing is shown. (H, W)
        ratio: The attention there, relative to uniform, full colour at four times
            or a quarter of it, NaN left grey. (H, W)

    Returns:
        image: The image. (H, W, 3)
    """
    lift = np.clip(np.log(np.maximum(ratio, 1e-12)) / np.log(4.0), -1, 1)
    lift = np.nan_to_num(lift)
    alpha = 0.9 * np.abs(lift)[..., None]
    colour = np.where(lift[..., None] > 0, (215, 48, 39), (69, 117, 180))
    rgb = (1 - alpha) * grey[..., None] * 255 + alpha * colour
    rgb[np.isnan(grey)] = EMPTY_RGB
    return rgb.astype(np.uint8)


def ratio_bar() -> np.ndarray:
    """Return the bar of the colours overlaid gives attention, on white.

    Returns:
        bar: The bar, four times uniform at the top, a quarter of it at the bottom.
            (ROW_HEIGHT, W, 3)
    """
    ratio = 4.0 ** np.linspace(1, -1, 256)[:, None]
    return colour_bar(overlaid(np.ones_like(ratio), ratio)[:, 0], ["4x", "1x", "0.25x"])


def attention_image(
    attention: Mapping[str, Mapping[str, np.ndarray]], query: str
) -> DrawnImage:
    """Return the query's original, masked, UMR and CMR above every other's attention.

    Args:
        attention: Each instrument's patches, as the evaluation drew them, the
            query's masked ones drawn only where it holds them.
        query: The instrument rebuilt.

    Returns:
        image: The query's rows, a spectrum's one per composite, above the
            attention row, each centred. (H, W, 3)
        caption: What the panels are, what is rebuilt from what, and each
            instrument's share per token.
    """
    arrays = attention[query]
    values = arrays["values"]
    shown = {"original": values}
    if "umr" in arrays:
        hidden = arrays["hidden"].reshape(-1, *[1] * (values.ndim - 1))
        shown |= {
            "input": np.where(hidden, np.nan, values),
            "UMR": np.where(hidden, arrays["umr"], values),
        }
    shown["CMR"] = arrays["cmr"]
    images = np.stack(
        [
            patch_mosaic(one, arrays["measured"], arrays["cell"])
            for one in shown.values()
        ]
    )  # (N, *image)
    tops = instrument_panels(query, images, arrays)
    sources = {name: held for name, held in attention.items() if name != query}
    grounds = [
        held for name, held in sources.items() if name != sharad.LAYOUT.instrument
    ]
    sounded = query == sharad.LAYOUT.instrument
    # One frame for every map: each ground patch, a ground query's too
    frame = map_frame(grounds if sounded else [*grounds, arrays])
    trace = observation_trace(arrays, frame, sounded)
    panels, drew = [], []
    total = sum(held["overall"].mean() for held in sources.values())
    for name, held in sources.items():
        drew.append(f"{name} {held['overall'].mean() / total:.0%}")
        if name == sharad.LAYOUT.instrument:
            ((_, grey, bar, ratio, identifier, share),) = sounder_passes(held, 1)
            drew[-1] += f" (pass {identifier} {share:.0%})"
        else:
            grey, bar, ratio = ground_map(held, frame)
        panel = overlaid(grey, ratio)
        if name != sharad.LAYOUT.instrument:
            # A track is drawn over what was measured alone
            panel[trace & ~np.isnan(grey) if sounded else trace] = TRACE_RGB
        panels.append(np.concatenate([panel, bar], axis=1))
    caption = f"{' | '.join(shown)}, {query} from {', '.join(sources)}: " + ", ".join(
        drew
    )
    return DrawnImage(
        centred_rows([*tops.values(), spaced_row([*panels, ratio_bar()])]), caption
    )
