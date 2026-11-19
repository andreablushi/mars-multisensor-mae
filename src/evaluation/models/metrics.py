"""Everything one view of the evaluated tiles comes to, and one scores table row."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class Score(NamedTuple):
    """One row of a scores table.

    Attributes:
        pooling: How the tokens were pooled.
        view: What was scored: an instrument, "all", a pair or a gallery.
        metric: The metric's name.
        value: What it came to.
    """

    pooling: str
    view: str
    metric: str
    value: float


@dataclass(frozen=True, slots=True)
class Metrics:
    """Everything one view of the tiles comes to.

    Attributes:
        pooling: How the tokens were pooled, CLS or GAP.
        view: The instrument, or "all".
        retrieval: Every tile's retrieval and ranking scores. (T,)
        knn: Accuracy and balanced accuracy of the kNN vote.
        classes: The classes, in the order the matrix holds them.
        distances: The mean cosine distance between the tiles of two classes. (C, C)
        silhouette: How much nearer each tile sits to its class than to the next. (T,)
        cross: How early the view finds the same tile in the others, averaged.
        rank: The RankMe effective rank of the view's raw training vectors and the
            most it can reach.
    """

    pooling: str
    view: str
    retrieval: dict[str, np.ndarray]
    knn: dict[str, float]
    classes: list[str]
    distances: np.ndarray
    silhouette: np.ndarray
    cross: dict[str, float]
    rank: dict[str, float]

    def scores(self) -> list[Score]:
        """Return every scalar the view comes to, as rows of the scores table.

        Returns:
            scores: mAP, the kNN scores, silhouette, cross-sensor mAP and RankMe.
        """
        return [
            Score(self.pooling, self.view, metric, value)
            for metric, value in {
                "mAP": float(self.retrieval["mAP"].mean()),
                **self.knn,
                "silhouette": float(self.silhouette.mean()),
                **self.cross,
                **self.rank,
            }.items()
        ]
