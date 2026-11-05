"""What the similarities between tiles come to against their classes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    silhouette_samples,
)


@dataclass(frozen=True, slots=True)
class Metrics:
    """Everything one view of the tiles comes to.

    Attributes:
        retrieval: Every tile's retrieval and ranking scores. (T,)
        knn: Accuracy and balanced accuracy of the kNN vote.
        classes: The classes, in the order the matrix holds them.
        distances: The mean cosine distance between the tiles of two classes. (C, C)
        silhouette: How much nearer each tile sits to its class than to the next. (T,)
    """

    retrieval: dict[str, np.ndarray]
    knn: dict[str, float]
    classes: list[str]
    distances: np.ndarray
    silhouette: np.ndarray


def metrics_by_view(
    vectors: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    classes: Mapping[str, str],
    k: int,
) -> dict[tuple[str, str], Metrics]:
    """Return the metrics of every pooling and view.

    Args:
        vectors: Each tile's vector, keyed by pooling, then view, then tile.
        classes: The class each tile earned, keyed by the tile.
        k: How many neighbours the kNN vote reads.

    Returns:
        metrics: Each view's metrics, keyed by its pooling and view.
    """
    metrics = {}
    for pooling, by_view in vectors.items():
        for view, by_tile in by_view.items():
            tiles = sorted(by_tile)
            held = np.array([by_tile[tile] for tile in tiles], np.float64)
            similarities = held @ held.T
            labels = [classes[tile] for tile in tiles]
            classed = sorted(set(labels))
            metrics[pooling, view] = Metrics(
                retrieval=retrieval_metrics(similarities, labels),
                knn=knn_metrics(similarities, labels, k),
                classes=classed,
                distances=class_distance_metrics(similarities, labels, classed),
                silhouette=silhouette_scores(similarities, labels),
            )
    return metrics


def ranked_neighbours(similarities: np.ndarray) -> np.ndarray:
    """Return every tile's other tiles, most similar first.

    Args:
        similarities: The cosine of every pair of tiles. (T, T)

    Returns:
        ranked: Each row's other tiles by decreasing similarity. (T, T-1)
    """
    held = similarities.copy()
    np.fill_diagonal(held, -np.inf)
    # A tile is never its own neighbour, so its own column, ranked last, is dropped.
    return np.argsort(-held, axis=1, kind="stable")[:, :-1]


def retrieval_metrics(
    similarities: np.ndarray, labels: Sequence[str]
) -> dict[str, np.ndarray]:
    """Return how early each tile's ranking reaches the tiles sharing its class.

    Args:
        similarities: The cosine of every pair of tiles. (T, T)
        labels: The class each tile carries, in the same order.

    Returns:
        scores: Every tile's average precision over the whole ranking. (T,)
    """
    held = np.asarray(labels)
    relevant = held[ranked_neighbours(similarities)] == held[:, None]  # (T, T-1)
    total = relevant.sum(axis=1)  # (T,)
    found = np.cumsum(relevant, axis=1)  # (T, T-1)
    # The precision at the rank of every classmate, zero at every other rank
    gained = found / np.arange(1, relevant.shape[1] + 1) * relevant  # (T, T-1)
    return {"mAP": gained.sum(axis=1) / total}


def knn_metrics(
    similarities: np.ndarray, labels: Sequence[str], k: int
) -> dict[str, float]:
    """Return how well the class each tile's k nearest tiles vote for matches its own.

    Args:
        similarities: The cosine of every pair of tiles. (T, T)
        labels: The class each tile carries, in the same order.
        k: How many neighbours vote, a tie going to the nearest.

    Returns:
        scores: Accuracy and balanced accuracy.
    """
    votes = np.asarray(labels)[ranked_neighbours(similarities)[:, :k]]  # (T, k)
    predicted = []
    for row in votes:
        counts = Counter(row)
        most = max(counts.values())
        predicted.append(next(one for one in row if counts[one] == most))
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced accuracy": float(balanced_accuracy_score(labels, predicted)),
    }


def class_distance_metrics(
    similarities: np.ndarray, labels: Sequence[str], classes: Sequence[str]
) -> np.ndarray:
    """Return how far each class stands from each, averaged over the tiles they hold.

    Args:
        similarities: The cosine of every pair of tiles. (T, T)
        labels: The class each tile carries, in the same order.
        classes: The classes, in the order the matrix holds them.

    Returns:
        matrix: The mean cosine distance between the tiles of two classes. (C, C)
    """
    held = np.asarray(labels)
    # A tile against itself says nothing, so it is left out of every mean.
    counted = np.where(np.eye(len(held), dtype=bool), np.nan, 1 - similarities)
    return np.array(
        [
            [
                np.nanmean(counted[np.ix_(held == one, held == other)])
                for other in classes
            ]
            for one in classes
        ]
    )


def silhouette_scores(similarities: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    """Return how much nearer each tile sits to its own class than to the next one.

    Args:
        similarities: The cosine of every pair of tiles. (T, T)
        labels: The class each tile carries, in the same order.

    Returns:
        scores: Every tile's silhouette over cosine distances, from -1 to 1. (T,)
    """
    # Rounding leaves the diagonal and near duplicates a hair off zero
    distances = np.clip(1 - similarities, 0, None)
    np.fill_diagonal(distances, 0)
    return silhouette_samples(distances, labels, metric="precomputed")
