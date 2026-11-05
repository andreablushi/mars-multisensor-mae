"""One instrument rebuilt from others whole, and where its CMR looked on each."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import numpy as np
from building.configs import sharad
from scipy.ndimage import binary_dilation

from logs.images.reconstruction import (
    COMPOSITES_NM,
    EMPTY_RGB,
    ROW_HEIGHT,
    SPACE,
    colour_bar,
    instrument_panels,
    nearest_bands,
    patch_mosaic,
    scaled_values,
    sounder_resampled,
    spaced_row,
    value_bar,
)

TRACE_RGB = (255, 140, 0)


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
        grey: The same, a spectrum's 770 nm albedo band as stored. (K, *ground)
        clip: The low and high to scale them by, NaN for an open side.
    """
    if patches.ndim < 4:
        return patches, arrays["clip"]
    band = nearest_bands(COMPOSITES_NM["albedo_770nm"])[0]
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one ground instrument's observations laid on a map, and its attention.

    Args:
        arrays: Its patches, as the evaluation drew them.
        frame: The map's north, south, west and east edges, in metres.

    Returns:
        grey: Its values from 0 to 1, NaN where none is measured, a spectrum's
            770 nm albedo band as stored. (ROW_HEIGHT, W)
        bar: The bar of those values. (ROW_HEIGHT, W', 3)
        ratio: The attention the tokens there drew on average, relative to uniform.
            (ROW_HEIGHT, W)
    """
    values, clip = grey_patches(arrays, arrays["values"])
    grey = laid_patches(arrays, values, frame)
    total, count = np.zeros((2, *grey.shape))
    for shown, patch, _ in ground_samples(arrays, map_places(frame)):
        total[shown] += arrays["attention"][patch]
        count[shown] += 1
    return (
        scaled_values(grey[None], clip)[0],
        value_bar(grey[None], clip),
        total / np.maximum(count, 1),
    )


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
        return area & binary_dilation(~area, np.ones((3, 3), bool), 3)
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
    return binary_dilation(line, np.ones((3, 3), bool), 1)


def sounder_passes(
    arrays: Mapping[str, np.ndarray], count: int
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, float]]:
    """Return the sounder's passes whose tokens drew the most attention each.

    Args:
        arrays: Its patches, as the evaluation drew them.
        count: How many passes are returned, the most attended first.

    Returns:
        passes: For each, which patches are its own (K,), its radargram from 0 to 1,
            NaN where none is measured, the bar of its values (ROW_HEIGHT, W, 3),
            and the attention its tokens drew relative to uniform, both maps
            resampled to ROW_HEIGHT rows (ROW_HEIGHT, ROW_HEIGHT / 2), the pass, and
            its share of the passes' attention per token.
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
    """Return a grey image coloured red where attention is above its mean, blue below.

    Args:
        grey: The image from 0 to 1, NaN where nothing is shown. (H, W)
        ratio: The attention there, relative to the instrument's mean, full colour
            at twice or half of it. (H, W)

    Returns:
        image: The image. (H, W, 3)
    """
    lift = np.clip(np.log(np.maximum(np.nan_to_num(ratio), 1e-12)) / np.log(2.0), -1, 1)
    alpha = 0.9 * np.abs(lift)[..., None]
    colour = np.where(lift[..., None] > 0, (215, 48, 39), (69, 117, 180))
    rgb = (1 - alpha) * grey[..., None] * 255 + alpha * colour
    rgb[np.isnan(grey)] = EMPTY_RGB
    return rgb.astype(np.uint8)


def ratio_bar() -> np.ndarray:
    """Return the bar of the colours overlaid gives attention, on white.

    Returns:
        bar: The bar, twice the mean at the top, half of it at the bottom.
            (ROW_HEIGHT, W, 3)
    """
    ratio = 2.0 ** np.linspace(1, -1, 256)[:, None]
    return colour_bar(overlaid(np.ones_like(ratio), ratio)[:, 0], ["2x", "1x", "0.5x"])


def attention_image(
    attention: Mapping[str, Mapping[str, np.ndarray]], query: str
) -> tuple[np.ndarray, str]:
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
    total = sum(held["attention"].mean() for held in sources.values())
    for name, held in sources.items():
        drew.append(f"{name} {held['attention'].mean() / total:.0%}")
        if name == sharad.LAYOUT.instrument:
            ((_, grey, bar, ratio, identifier, share),) = sounder_passes(held, 1)
            drew[-1] += f" (pass {identifier} {share:.0%})"
        else:
            grey, bar, ratio = ground_map(held, frame)
        # Where it looked within the instrument, its share being in the caption
        panel = overlaid(grey, ratio / held["attention"].mean())
        if name != sharad.LAYOUT.instrument:
            # A track is drawn over what was measured alone
            panel[trace & ~np.isnan(grey) if sounded else trace] = TRACE_RGB
        panels.append(np.concatenate([panel, bar], axis=1))
    caption = f"{' | '.join(shown)}, {query} from {', '.join(sources)}: " + ", ".join(
        drew
    )
    return centred_rows([*tops.values(), spaced_row([*panels, ratio_bar()])]), caption


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
