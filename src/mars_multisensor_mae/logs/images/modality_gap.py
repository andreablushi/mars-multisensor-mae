"""Every instrument's tile vectors on the first two principal components of them all."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

from mars_multisensor_mae.logs.images.panels import DrawnImage


def modality_gap_image(instruments: Mapping[str, np.ndarray]) -> DrawnImage:
    """Return one pooling's instrument vectors on PC1 and PC2, one colour each.

    Args:
        instruments: Each instrument's unit tile vectors. (T, D)

    Returns:
        image: The instruments' clouds on one pair of axes. (H, W, 3)
        caption: The share of the variance PC1 and PC2 explain.
    """
    pca = PCA(2).fit(np.concatenate(list(instruments.values())))
    first, second = pca.explained_variance_ratio_
    figure = Figure(figsize=(6, 6), dpi=120)
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_subplot()
    for name, held in instruments.items():
        axes.scatter(*pca.transform(held).T, s=8, alpha=0.6, linewidths=0, label=name)
    axes.set_xlabel(f"PC1 ({first:.1%})")
    axes.set_ylabel(f"PC2 ({second:.1%})")
    axes.legend(markerscale=2)
    figure.tight_layout()
    canvas.draw()
    caption = f"PC1 {first:.1%}, PC2 {second:.1%} of the variance"
    return DrawnImage(np.asarray(canvas.buffer_rgba())[..., :3].copy(), caption)
