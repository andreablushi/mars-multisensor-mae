"""One instrument rebuilt from others whole, and where its CMR looked on each."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

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
SQUARE = np.ones((3, 3), bool)
QUERY_COMPOSITE = "false_colour_2529_1506_1080nm"
GREY_COMPOSITE = "albedo_770nm"
TOP_PASSES = 3


def map_frame(framed: Sequence[Mapping[str, np.ndarray]]) -> np.ndarray:
    """Return a map's edges around ground patches, half the widest patch past them.

    Args:
        framed: Each ground instrument's patches, as the evaluation drew them.

    Returns:
        frame: The map's north, south, west and east edges, in metres.
    """
    half = max(
        (np.array(held["values"].shape[1:3]) * held["spacing"]).max() / 2
        for held in framed
    )
    centres = np.concatenate([held["position"][:, :2] for held in framed])
    (west, south), (east, north) = (
        centres.min(axis=0) - half,
        centres.max(axis=0) + half,
    )
    return np.array([north, south, west, east])


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
        arrays: One ground instrument's patches, as the evaluation drew them.
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


def grey_patches(
    arrays: Mapping[str, np.ndarray], patches: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return one ground instrument's patches as one grey value a sample.

    Args:
        arrays: Its patches, as the evaluation drew them.
        patches: What is drawn of them, as z-scores. (K, *P)

    Returns:
        grey: The same, a spectrum's GREY_COMPOSITE band as stored. (K, *ground)
        clip: The low and high to scale them by, NaN for an open side.
    """
    if patches.ndim < 4:
        return patches, arrays["clip"]
    band = nearest_bands(COMPOSITES_NM[GREY_COMPOSITE])[0]
    return patches[..., band] * arrays["std"][band] + arrays["mean"][band], np.full(
        2, np.nan
    )


def laid_patches(
    arrays: Mapping[str, np.ndarray], patches: np.ndarray, frame: np.ndarray
) -> np.ndarray:
    """Return one value a sample of one ground instrument laid on a map.

    Args:
        arrays: Its patches, as the evaluation drew them.
        patches: One value per sample of each patch. (K, *ground)
        frame: The map's north, south, west and east edges, in metres.

    Returns:
        laid: The values, the most measured observation on top, NaN where none is
            measured. (ROW_HEIGHT, W)
    """
    places = map_places(frame)
    laid = np.full(places.shape[:2], np.nan)
    for shown, patch, inner in ground_samples(arrays, places):
        laid[shown] = patches[patch, *inner.T]
    return laid


def ground_map(
    arrays: Mapping[str, np.ndarray], frame: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return one ground instrument's observations laid on a map, and its attention.

    Args:
        arrays: Its patches, as the evaluation drew them.
        frame: The map's north, south, west and east edges, in metres.

    Returns:
        grey: Its values from 0 to 1, NaN where none is measured, a spectrum's
            GREY_COMPOSITE band as stored. (ROW_HEIGHT, W)
        ratio: The attention the tokens there drew on average, relative to uniform.
            (ROW_HEIGHT, W)
    """
    values, clip = grey_patches(arrays, arrays["values"])
    grey = laid_patches(arrays, values, frame)
    total, count = np.zeros((2, *grey.shape))
    for shown, patch, _ in ground_samples(arrays, map_places(frame)):
        total[shown] += arrays["attention"][patch]
        count[shown] += 1
    return scaled_values(grey[None], clip)[0], total / np.maximum(count, 1)


def observation_trace(
    arrays: Mapping[str, np.ndarray], frame: np.ndarray, sounder: bool
) -> np.ndarray:
    """Return where one observation lies on a map ROW_HEIGHT rows high.

    Args:
        arrays: Its patches, as the evaluation drew them.
        frame: The map's north, south, west and east edges, in metres.
        sounder: Whether it is a sounder's pass, which lies on a line.

    Returns:
        trace: The outline of the ground it measured, or a sounder's track.
            (ROW_HEIGHT, W)
    """
    places = map_places(frame)
    if not sounder:
        area = np.zeros(places.shape[:2], bool)
        for shown, _, _ in ground_samples(arrays, places):
            area |= shown
        return area & binary_dilation(~area, SQUARE, OUTLINE)
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
    return binary_dilation(line, SQUARE, OUTLINE // 2)


def sounder_passes(
    arrays: Mapping[str, np.ndarray], count: int
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, str, float]]:
    """Return the sounder's passes whose tokens drew the most attention each.

    Args:
        arrays: Its patches, as the evaluation drew them.
        count: How many passes are returned, the most attended first.

    Returns:
        passes: For each, which patches are its own (K,), its radargram from 0 to 1,
            NaN where none is measured, and the attention its tokens drew relative
            to uniform, both resampled to ROW_HEIGHT rows (ROW_HEIGHT, ROW_HEIGHT /
            2), the pass, and its share of the passes' attention per token.
    """
    by_pass = np.bincount(arrays["observation"], weights=arrays["attention"])
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
            ROW_HEIGHT,
        )
        passes.append(
            (
                mine,
                scaled_values(grey[None], arrays["clip"])[0],
                ratio,
                str(arrays["identifier"][chosen]),
                by_pass[chosen] / by_pass.sum(),
            )
        )
    return passes


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
        attention: Each instrument's patches, as the evaluation drew them.
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
    # One frame for every map: each ground patch, a ground query's too
    frame = map_frame(grounds if query == SOUNDER else [*grounds, arrays])
    trace = observation_trace(arrays, frame, query == SOUNDER)
    panels, drew = [], []
    total = sum(held["attention"].mean() for held in sources.values())
    for name, held in sources.items():
        drew.append(f"{name} {held['attention'].mean() / total:.0%}")
        if name == SOUNDER:
            ((_, grey, ratio, identifier, share),) = sounder_passes(held, 1)
            drew[-1] += f" (pass {identifier} {share:.0%})"
        else:
            grey, ratio = ground_map(held, frame)
        # Where it looked within the instrument, its share being in the caption
        panel = overlaid(grey, ratio / held["attention"].mean())
        if name != SOUNDER:
            # A track is drawn over what was measured alone
            panel[trace & ~np.isnan(grey) if query == SOUNDER else trace] = TRACE_RGB
        panels.append(panel)
    cls = sum(held["cls"] for held in sources.values())
    caption = (
        f"{query} original | CMR from {', '.join(sources)} whole. Attention per"
        " token: "
        + ", ".join(drew)
        + f"; the CLS tokens drew {cls:.0%} of all. Red above the instrument's mean,"
        f" blue below, full at {FULL_RATIO:g}x. Orange: where the {query}"
        " observation lies."
    )
    return centred_rows([top, spaced_row(panels)]), caption


def sounder_read_image(
    attention: Mapping[str, Mapping[str, np.ndarray]], query: str
) -> tuple[np.ndarray, str]:
    """Return a ground query's original and CMR from the sounder alone, and its passes.

    Args:
        attention: The query's and the sounder's patches, as the evaluation drew them.
        query: The ground instrument rebuilt.

    Returns:
        image: The query's two panels with the most attended tracks, above those
            passes' radargrams, each row centred. (H, W, 3)
        caption: What each panel shows and what each pass drew.
    """
    arrays, held = attention[query], attention[SOUNDER]
    frame = map_frame([arrays])
    laid = []
    for one in (arrays["values"], arrays["cmr"]):
        grey, clip = grey_patches(arrays, one)
        laid.append(laid_patches(arrays, grey, frame))
    passes = sounder_passes(held, TOP_PASSES)
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
    radargrams = [
        overlaid(grey, ratio / held["attention"].mean())
        for _, grey, ratio, *_ in passes
    ]
    caption = (
        f"{query} original | CMR from {SOUNDER} alone, orange: the tracks of the"
        f" {len(passes)} passes read most per token. Below, left to right: "
        + ", ".join(
            f"pass {identifier} {share:.0%}" for *_, identifier, share in passes
        )
        + f" of the passes' attention per token; the CLS drew {held['cls']:.0%} of all."
        f" Red above the {SOUNDER} mean, blue below, full at {FULL_RATIO:g}x."
    )
    return centred_rows([spaced_row(top), spaced_row(radargrams)]), caption


def centred_rows(rows: list[np.ndarray]) -> np.ndarray:
    """Return rows ROW_HEIGHT high stacked and centred on white, a gap around each.

    Args:
        rows: The rows, top first. (ROW_HEIGHT, W, 3) each

    Returns:
        image: The rows. (H, W', 3)
    """
    width, gap = max(row.shape[1] for row in rows), ROW_HEIGHT // SPACE
    image = np.full(
        ((len(rows) + 1) * gap + len(rows) * ROW_HEIGHT, width, 3), 255, np.uint8
    )
    for at, row in enumerate(rows):
        left = (width - row.shape[1]) // 2
        top_row = gap + at * (ROW_HEIGHT + gap)
        image[top_row : top_row + ROW_HEIGHT, left : left + row.shape[1]] = row
    return image
