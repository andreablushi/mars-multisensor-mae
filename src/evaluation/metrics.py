"""What the similarities between tiles come to against their classes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from umap import UMAP

TOP_K = (1, 5, 10, 20)


def ranked_neighbours(similarities: np.ndarray) -> np.ndarray:
    """Return every tile's other tiles, most similar first.

    Args:
        similarities: The cosine of every pair of tiles, NaN where not comparable.
            (T, T)

    Returns:
        ranked: Each row's other tiles by decreasing similarity, the incomparable
            last. (T, T-1)
    """
    unreachable = np.eye(len(similarities), dtype=bool) | np.isnan(similarities)
    held = np.where(unreachable, -np.inf, similarities)
    # A tile is never its own neighbour, so its own column, ranked last, is dropped.
    return np.argsort(-held, axis=1, kind="stable")[:, :-1]


def retrieval_by_tile(
    similarities: np.ndarray, labels: Sequence[str]
) -> dict[str, np.ndarray]:
    """Return how far each tile's nearest tiles share its class.

    Args:
        similarities: The cosine of every pair of tiles. (T, T)
        labels: The class each tile carries, in the same order.

    Returns:
        scores: Every tile's precision, recall and F1 at every k of TOP_K. (T,)
    """
    held = np.asarray(labels)
    relevant = held[ranked_neighbours(similarities)] == held[:, None]  # (T, T-1)
    total = relevant.sum(axis=1)  # (T,)
    scores = {}
    for k in TOP_K:
        taken = relevant[:, :k]  # (T, k)
        precision = taken.mean(axis=1)  # (T,)
        recall = taken.sum(axis=1) / total  # (T,)
        together = np.maximum(precision + recall, np.finfo(float).eps)
        scores |= {
            f"precision@{k}": precision,
            f"recall@{k}": recall,
            f"f1@{k}": 2 * precision * recall / together,
        }
    return scores


def class_distances(
    similarities: np.ndarray, labels: Sequence[str]
) -> tuple[list[str], np.ndarray]:
    """Return how far each class stands from each, averaged over the tiles they hold.

    Args:
        similarities: The cosine of every pair of tiles. (T, T)
        labels: The class each tile carries, in the same order.

    Returns:
        classes: The classes, in the order the matrix holds them.
        matrix: The mean cosine distance between the tiles of two classes. (C, C)
    """
    held = np.asarray(labels)
    classes = sorted(set(labels))
    # A tile against itself says nothing, so it is left out of every mean.
    counted = np.where(np.eye(len(held), dtype=bool), np.nan, 1 - similarities)
    matrix = np.array(
        [
            [
                np.nanmean(counted[np.ix_(held == one, held == other)])
                for other in classes
            ]
            for one in classes
        ]
    )
    return classes, matrix


def knn_predictions(
    similarities: np.ndarray, labels: Sequence[str], k: int
) -> np.ndarray:
    """Return the class each tile's k nearest tiles vote for.

    Args:
        similarities: The cosine of every pair of tiles. (T, T)
        labels: The class each tile carries, in the same order.
        k: How many neighbours vote.

    Returns:
        predicted: The most voted class, a tie going to the nearest of them. (T,)
    """
    votes = np.asarray(labels)[ranked_neighbours(similarities)[:, :k]]  # (T, k)
    predicted = []
    for row in votes:
        counts = Counter(row)
        most = max(counts.values())
        predicted.append(next(one for one in row if counts[one] == most))
    return np.array(predicted)


def knn_scores(labels: Sequence[str], predicted: Sequence[str]) -> dict[str, float]:
    """Return how well the voted classes match the true ones.

    Args:
        labels: The class each tile carries.
        predicted: The class its neighbours voted for, in the same order.

    Returns:
        scores: Accuracy, balanced accuracy, macro F1 and the F1 of every class.
    """
    classes = sorted(set(labels))
    per_class = f1_score(labels, predicted, labels=classes, average=None)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced accuracy": float(balanced_accuracy_score(labels, predicted)),
        "macro f1": float(f1_score(labels, predicted, average="macro")),
    } | {
        f"f1/{name}": float(score)
        for name, score in zip(classes, per_class, strict=True)
    }


def knn_confusion(
    labels: Sequence[str], predicted: Sequence[str]
) -> tuple[list[str], np.ndarray]:
    """Return how many tiles of each class were voted into each.

    Args:
        labels: The class each tile carries.
        predicted: The class its neighbours voted for, in the same order.

    Returns:
        classes: The classes, in the order the matrix holds them.
        matrix: Tiles of the row's class voted into the column's. (C, C)
    """
    classes = sorted(set(labels))
    return classes, confusion_matrix(labels, predicted, labels=classes)


def umap_projection(similarities: np.ndarray, seed: int) -> np.ndarray:
    """Return every tile laid on a plane by UMAP, read off the cosine distances.

    Args:
        similarities: The cosine of every pair of tiles, NaN where not comparable.
            (T, T)
        seed: What the layout is drawn with, so the same similarities lay out the same.

    Returns:
        projection: Where each tile lands on the plane, in the same order. (T, 2)
    """
    distances = np.nan_to_num(np.clip(1 - similarities, 0, 2), nan=2.0)
    np.fill_diagonal(distances, 0.0)
    return UMAP(metric="precomputed", random_state=seed).fit_transform(distances)
