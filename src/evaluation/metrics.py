"""What the similarities between tiles come to against their classes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    silhouette_samples,
)

from evaluation.results import Metrics


def cosine_similarity(vectors: np.ndarray) -> np.ndarray:
    """Return the cosine between every pair of tiles' vectors.

    Args:
        vectors: One unit vector per tile. (T, D)

    Returns:
        similarities: The cosine of every pair. (T, T)
    """
    return vectors @ vectors.T


def compute_metrics(similarities: np.ndarray, labels: Sequence[str], k: int) -> Metrics:
    """Return every metric of one view.

    Args:
        similarities: The cosine of every pair of tiles. (T, T)
        labels: The class of each tile, in the same order.
        k: How many neighbours the kNN vote reads.

    Returns:
        metrics: The view's metrics.
    """
    classes = sorted(set(labels))
    knn, confusion = knn_metrics(similarities, labels, classes, k)
    return Metrics(
        labels=list(labels),
        retrieval=retrieval_metrics(similarities, labels),
        knn=knn,
        classes=classes,
        confusion=confusion,
        distances=class_distance_metrics(similarities, labels, classes),
        silhouette=silhouette_scores(similarities, labels),
    )


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
    similarities: np.ndarray, labels: Sequence[str], classes: Sequence[str], k: int
) -> tuple[dict[str, float], np.ndarray]:
    """Return how well the class each tile's k nearest tiles vote for matches its own.

    Args:
        similarities: The cosine of every pair of tiles. (T, T)
        labels: The class each tile carries, in the same order.
        classes: The classes, in the order the matrix holds them.
        k: How many neighbours vote.

    Returns:
        scores: Accuracy and balanced accuracy.
        confusion: Tiles of the row's class voted into the column's, a tie going to
            the nearest. (C, C)
    """
    votes = np.asarray(labels)[ranked_neighbours(similarities)[:, :k]]  # (T, k)
    predicted = []
    for row in votes:
        counts = Counter(row)
        most = max(counts.values())
        predicted.append(next(one for one in row if counts[one] == most))
    scores = {
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced accuracy": float(balanced_accuracy_score(labels, predicted)),
    }
    return scores, confusion_matrix(labels, predicted, labels=classes)


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
