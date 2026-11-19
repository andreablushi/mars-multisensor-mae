"""Every instrument's tile vectors on the first two principal components of them all."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.decomposition import PCA


def modality_gap_image(
    by_view: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[np.ndarray, str]:
    """Return one pooling's instrument vectors on PC1 and PC2, one colour each.

    Args:
        by_view: Each tile's unit vector, keyed by view, then tile; "all" is left out.

    Returns:
        image: The instruments' clouds on one pair of axes. (H, W, 3)
        caption: The share of the variance PC1 and PC2 explain.
    """
    stacked = {
        view: np.array(list(by_tile.values()))
        for view, by_tile in by_view.items()
        if view != "all"
    }
    pca = PCA(2).fit(np.concatenate(list(stacked.values())))
    first, second = pca.explained_variance_ratio_
    figure = Figure(figsize=(6, 6), dpi=120)
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_subplot()
    for view, held in stacked.items():
        axes.scatter(*pca.transform(held).T, s=8, alpha=0.6, linewidths=0, label=view)
    axes.set_xlabel(f"PC1 ({first:.1%})")
    axes.set_ylabel(f"PC2 ({second:.1%})")
    axes.legend(markerscale=2)
    figure.tight_layout()
    canvas.draw()
    caption = f"PC1 {first:.1%}, PC2 {second:.1%} of the variance"
    return np.asarray(canvas.buffer_rgba())[..., :3].copy(), caption
