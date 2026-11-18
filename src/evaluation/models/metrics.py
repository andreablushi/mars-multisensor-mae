"""Everything one view of the evaluated tiles comes to."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Metrics:
    """Everything one view of the tiles comes to.

    Attributes:
        retrieval: Every tile's retrieval and ranking scores. (T,)
        knn: Accuracy and balanced accuracy of the kNN vote.
        classes: The classes, in the order the matrix holds them.
        distances: The mean cosine distance between the tiles of two classes. (C, C)
        silhouette: How much nearer each tile sits to its class than to the next. (T,)
        cross: How early the view finds the same tile in the others, averaged.
        rank: The RankMe effective rank of the view's raw training vectors and the
            most it can reach.
    """

    retrieval: dict[str, np.ndarray]
    knn: dict[str, float]
    classes: list[str]
    distances: np.ndarray
    silhouette: np.ndarray
    cross: dict[str, float]
    rank: dict[str, float]
