"""What the similarities between tiles come to against their classes."""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from umap import UMAP

from evaluation.results import Metrics

TOP_K = (1, 5, 10, 20)


def compute_metrics(
    similarities: np.ndarray,
    tiles: Sequence[str],
    labels: Sequence[str],
    k: int,
    seed: int,
) -> Metrics:
    """Return every metric of one view.

    Args:
        similarities: The cosine of every pair of tiles. (T, T)
        tiles: The tiles, in the same order.
        labels: The class of each, in the same order.
        k: How many neighbours the kNN vote reads.
        seed: What the UMAP layout is drawn with.

    Returns:
        metrics: The view's metrics.
    """
    classes = sorted(set(labels))
    knn, confusion = knn_metrics(similarities, labels, classes, k)
    return Metrics(
        tiles=list(tiles),
        labels=list(labels),
        retrieval=retrieval_metrics(similarities, labels),
        knn=knn,
        classes=classes,
        confusion=confusion,
        distances=class_distance_metrics(similarities, labels, classes),
        projection=umap_projection(similarities, seed),
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
    """Return how far each tile's nearest tiles share its class, and how early.

    Args:
        similarities: The cosine of every pair of tiles. (T, T)
        labels: The class each tile carries, in the same order.

    Returns:
        scores: Every tile's precision, recall and F1 at every k of TOP_K, its
            average precision over the whole ranking and at every k, the
            reciprocal rank of its first classmate and its R-precision. (T,)
    """
    held = np.asarray(labels)
    relevant = held[ranked_neighbours(similarities)] == held[:, None]  # (T, T-1)
    total = relevant.sum(axis=1)  # (T,)
    found = np.cumsum(relevant, axis=1)  # (T, T-1)
    # The precision at the rank of every classmate, zero at every other rank
    gained = found / np.arange(1, relevant.shape[1] + 1) * relevant  # (T, T-1)
    scores = {
        "mAP": gained.sum(axis=1) / total,
        "MRR": 1 / (relevant.argmax(axis=1) + 1),
        "R-precision": found[np.arange(len(total)), total - 1] / total,
    }
    for k in TOP_K:
        taken = relevant[:, :k]  # (T, k)
        precision = taken.mean(axis=1)  # (T,)
        recall = taken.sum(axis=1) / total  # (T,)
        together = np.maximum(precision + recall, np.finfo(float).eps)
        scores |= {
            f"precision@{k}": precision,
            f"recall@{k}": recall,
            f"f1@{k}": 2 * precision * recall / together,
            f"mAP@{k}": gained[:, :k].sum(axis=1) / np.minimum(total, k),
        }
    return scores


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


def umap_projection(similarities: np.ndarray, seed: int) -> np.ndarray:
    """Return every tile laid on a plane by UMAP, read off the cosine distances.

    Args:
        similarities: The cosine of every pair of tiles. (T, T)
        seed: What the layout is drawn with, so the same similarities lay out the same.

    Returns:
        projection: Where each tile lands on the plane, in the same order. (T, 2)
    """
    distances = np.clip(1 - similarities, 0, 2)
    np.fill_diagonal(distances, 0.0)
    with warnings.catch_warnings():
        # No inverse transform is ever asked of the layout
        warnings.filterwarnings("ignore", "using precomputed metric")
        # A seeded layout runs on one job either way
        return UMAP(metric="precomputed", random_state=seed, n_jobs=1).fit_transform(
            distances
        )
