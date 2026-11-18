"""Laying ground patches on one map frame, and tracing an observation on it."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import numpy as np
from scipy.ndimage import binary_dilation

from logs.images.panels import COMPOSITES_NM, ROW_HEIGHT, nearest_bands


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
