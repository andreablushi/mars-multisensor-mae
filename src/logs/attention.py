"""One instrument rebuilt from the others whole, and where its CMR looked on each."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import numpy as np
from PIL import Image

from logs.reconstruction import (
    COMPOSITES_NM,
    EMPTY_RGB,
    SOUNDER,
    SPACE,
    instrument_panels,
    nearest_bands,
    patch_mosaic,
    scaled_values,
    sounder_resampled,
    spaced_row,
)

ROW_HEIGHT = 512
ABOVE_RGB = (215, 48, 39)
BELOW_RGB = (69, 117, 180)
OPACITY = 0.9
FULL_RATIO = 2.0
TRACE_RGB = (255, 140, 0)
OUTLINE = 3
QUERY_COMPOSITE = "false_colour_2529_1506_1080nm"
GREY_COMPOSITE = "albedo_770nm"


def map_places(frame: np.ndarray) -> np.ndarray:
    """Return where every pixel centre of a map ROW_HEIGHT rows high lies.

    Args:
        frame: The map's north, south, west and east edges, in metres.

    Returns:
        places: Each pixel's north and east, in metres. (ROW_HEIGHT, W, 2)
    """
    north, south, west, east = frame
    step = (north - south) / ROW_HEIGHT
    return np.stack(
        np.meshgrid(
            north - (np.arange(round((north - south) / step)) + 0.5) * step,
            west + (np.arange(round((east - west) / step)) + 0.5) * step,
            indexing="ij",
        ),
        axis=-1,
    )


def ground_samples(
    arrays: Mapping[str, np.ndarray], places: np.ndarray
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield which measured sample of each observation lands on each pixel of a map.

    Args:
        arrays: One ground instrument's patches, as attend wrote them.
        places: Where every pixel centre of the map lies. (H, W, 2)

    Yields:
        shown: Which pixels a measured sample of the observation lands on. (H, W)
        patch: The patch each of those comes from. (M,)
        inner: Where in that patch, its row and column. (M, 2)
    """
    ground = np.array(arrays["values"].shape[1:3])
    measured = arrays["measured"].reshape(-1, *ground)
    cells = arrays["cell"][:, :2]
    centres = arrays["position"][:, [1, 0]]  # (K, 2) north, east
    # The least measured first, so the most measured is laid last
    for observation in np.unique(arrays["observation"]):
        mine = np.flatnonzero(arrays["observation"] == observation)
        # Its grid of patches, fitted to their centres
        fit, *_ = np.linalg.lstsq(
            np.c_[cells[mine], np.ones(mine.size)], centres[mine], rcond=None
        )  # (3, 2)
        sample = np.floor(
            ((places - fit[2]) @ np.linalg.pinv(fit[:2]) + 0.5) * ground
        ).astype(int)  # (H, W, 2)
        # Each patch on its grid, bordered by -1 that every pixel off it clips to
        index = np.full(cells[mine].max(axis=0) + 3, -1)
        index[tuple(cells[mine].T + 1)] = mine
        cell = np.clip(sample // ground + 1, 0, np.array(index.shape) - 1)
        patch = index[cell[..., 0], cell[..., 1]]
        inner = sample % ground
        shown = patch >= 0
        shown[shown] = measured[patch[shown], *inner[shown].T]
        yield shown, patch[shown], inner[shown]


def ground_map(
    arrays: Mapping[str, np.ndarray], frame: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return one ground instrument's observations laid on a map, and its attention.

    Args:
        arrays: Its patches, as attend wrote them.
        frame: The map's north, south, west and east edges, in metres.

    Returns:
        grey: Its values from 0 to 1, NaN where none is measured, a spectrum's
            GREY_COMPOSITE band as stored. (ROW_HEIGHT, W)
        ratio: The attention the tokens there drew on average, relative to uniform.
            (ROW_HEIGHT, W)
    """
    values, clip = arrays["values"], arrays["clip"]
    if values.ndim == 4:
        band = nearest_bands(COMPOSITES_NM[GREY_COMPOSITE])[0]
        values = values[..., band] * arrays["std"][band] + arrays["mean"][band]
        clip = np.full(2, np.nan)
    places = map_places(frame)
    grey = np.full(places.shape[:2], np.nan)
    total, count = np.zeros((2, *grey.shape))
    for shown, patch, inner in ground_samples(arrays, places):
        grey[shown] = values[patch, *inner.T]
        total[shown] += arrays["attention"][patch]
        count[shown] += 1
    return scaled_values(grey[None], clip)[0], total / np.maximum(count, 1)


def thickened(mask: np.ndarray, width: int) -> np.ndarray:
    """Return a mask grown by some pixels every way.

    Args:
        mask: The mask. (H, W)
        width: How many pixels it grows by.

    Returns:
        grown: The mask, grown. (H, W)
    """
    grown = mask.copy()
    for _ in range(width):
        grown[1:] |= grown[:-1].copy()
        grown[:-1] |= grown[1:].copy()
        grown[:, 1:] |= grown[:, :-1].copy()
        grown[:, :-1] |= grown[:, 1:].copy()
    return grown


def query_trace(
    arrays: Mapping[str, np.ndarray], frame: np.ndarray, sounder: bool
) -> np.ndarray:
    """Return where the query's observation lies on a map ROW_HEIGHT rows high.

    Args:
        arrays: The query's patches, as attend wrote them.
        frame: The map's north, south, west and east edges, in metres.
        sounder: Whether the query is a sounder, which lies on a line.

    Returns:
        trace: The outline of the ground it measured, or a sounder's track.
            (ROW_HEIGHT, W)
    """
    places = map_places(frame)
    if not sounder:
        area = np.zeros(places.shape[:2], bool)
        for shown, _, _ in ground_samples(arrays, places):
            area |= shown
        return area & thickened(~area, OUTLINE)
    # The track through its patch centres, one per trace pair, densely sampled
    _, first = np.unique(arrays["cell"][:, -1], return_index=True)
    track = arrays["position"][first, :2]
    points = np.concatenate(
        [track] + [np.linspace(one, two, 32) for one, two in zip(track, track[1:])]
    )
    # Each point's row from the north edge, its column from the west
    step = (frame[0] - frame[1]) / ROW_HEIGHT
    pixels = np.floor(
        np.c_[frame[0] - points[:, 1], points[:, 0] - frame[2]] / step
    ).astype(int)
    inside = ((pixels >= 0) & (pixels < places.shape[:2])).all(axis=-1)
    line = np.zeros(places.shape[:2], bool)
    line[tuple(pixels[inside].T)] = True
    return thickened(line, OUTLINE // 2)


def sounder_pass(
    arrays: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, str, float]:
    """Return the sounder's pass that drew the most attention, and that attention.

    Args:
        arrays: Its patches, as attend wrote them.

    Returns:
        grey: The pass's radargram from 0 to 1, NaN where none is measured,
            resampled to ROW_HEIGHT rows. (ROW_HEIGHT, ROW_HEIGHT / 2)
        ratio: The attention its tokens drew, relative to uniform, likewise.
        identifier: The pass.
        share: Its share of what the sounder's tokens drew.
    """
    by_pass = np.bincount(arrays["observation"], weights=arrays["attention"])
    chosen = by_pass.argmax()
    mine = arrays["observation"] == chosen
    values = arrays["values"][mine]
    attention = np.broadcast_to(
        arrays["attention"][mine].reshape(-1, *[1] * (values.ndim - 1)), values.shape
    )
    grey, ratio = sounder_resampled(
        np.stack(
            [
                patch_mosaic(one, arrays["measured"][mine], arrays["cell"][mine])
                for one in (values, attention)
            ]
        ),
        ROW_HEIGHT,
    )
    return (
        scaled_values(grey[None], arrays["clip"])[0],
        ratio,
        str(arrays["identifier"][chosen]),
        by_pass[chosen] / by_pass.sum(),
    )


def overlaid(grey: np.ndarray, ratio: np.ndarray) -> np.ndarray:
    """Return a grey image coloured red where attention is above its mean, blue below.

    Args:
        grey: The image from 0 to 1, NaN where nothing is shown. (H, W)
        ratio: The attention there, relative to the instrument's mean, full colour
            at FULL_RATIO or its inverse. (H, W)

    Returns:
        image: The image. (H, W, 3)
    """
    lift = np.clip(
        np.log(np.maximum(np.nan_to_num(ratio), 1e-12)) / np.log(FULL_RATIO), -1, 1
    )
    alpha = OPACITY * np.abs(lift)[..., None]
    colour = np.where(lift[..., None] > 0, ABOVE_RGB, BELOW_RGB)
    rgb = (1 - alpha) * grey[..., None] * 255 + alpha * colour
    rgb[np.isnan(grey)] = EMPTY_RGB
    return rgb.astype(np.uint8)


def attention_image(
    attention: Mapping[str, Mapping[str, np.ndarray]], query: str
) -> tuple[np.ndarray, str]:
    """Return the query's original and CMR above every other instrument's attention.

    Args:
        attention: Each instrument's patches, as attend wrote them.
        query: The instrument rebuilt.

    Returns:
        image: The two rows, each centred. (H, W, 3)
        caption: What each panel shows and what each instrument drew.
    """
    arrays = attention[query]
    images = np.stack(
        [
            patch_mosaic(one, arrays["measured"], arrays["cell"])
            for one in (arrays["values"], arrays["cmr"])
        ]
    )  # (2, *image)
    (top,) = instrument_panels(
        query, images, arrays, {QUERY_COMPOSITE: COMPOSITES_NM[QUERY_COMPOSITE]}
    ).values()
    top = np.array(
        Image.fromarray(top).resize(
            (round(top.shape[1] * ROW_HEIGHT / top.shape[0]), ROW_HEIGHT),
            Image.Resampling.NEAREST,
        )
    )
    sources = {name: held for name, held in attention.items() if name != query}
    grounds = [held for name, held in sources.items() if name != SOUNDER]
    # One frame for every map: each ground patch centre, a ground query's too, half
    # the widest patch around
    half = max(
        (np.array(held["values"].shape[1:3]) * held["spacing"]).max() / 2
        for held in grounds
    )
    framed = grounds if query == SOUNDER else [*grounds, arrays]
    centres = np.concatenate([held["position"][:, :2] for held in framed])
    (west, south), (east, north) = (
        centres.min(axis=0) - half,
        centres.max(axis=0) + half,
    )
    frame = np.array([north, south, west, east])
    trace = query_trace(arrays, frame, query == SOUNDER)
    panels, drew = [], []
    for name, held in sources.items():
        drew.append(f"{name} {held['share'] - held['cls']:.0%}")
        if name == SOUNDER:
            grey, ratio, identifier, share = sounder_pass(held)
            drew[-1] += f" (pass {identifier} drew {share:.0%} of it)"
        else:
            grey, ratio = ground_map(held, frame)
        # Where it looked within the instrument, its share being in the caption
        panel = overlaid(grey, ratio / held["attention"].mean())
        if name != SOUNDER:
            # A track is drawn over what was measured alone
            panel[trace & ~np.isnan(grey) if query == SOUNDER else trace] = TRACE_RGB
        panels.append(panel)
    # Both rows centred on white, a gap above, between and below
    rows = [top, spaced_row(panels)]
    width, gap = max(row.shape[1] for row in rows), ROW_HEIGHT // SPACE
    image = np.full((3 * gap + 2 * ROW_HEIGHT, width, 3), 255, np.uint8)
    for at, row in enumerate(rows):
        left = (width - row.shape[1]) // 2
        top_row = gap + at * (ROW_HEIGHT + gap)
        image[top_row : top_row + ROW_HEIGHT, left : left + row.shape[1]] = row
    cls = sum(held["cls"] for held in sources.values())
    caption = (
        f"{query} original | CMR from the others whole. Attention: "
        + ", ".join(drew)
        + f", CLS {cls:.0%}. Red above the instrument's mean, blue below, full at"
        f" {FULL_RATIO:g}x. Orange: where the {query} observation lies."
    )
    return image, caption
