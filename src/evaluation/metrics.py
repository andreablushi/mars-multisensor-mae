"""What a distance between tiles comes to, read against the classes they carry."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import silhouette_samples


def retrieval_metrics(
    distances: np.ndarray, labels: Sequence[str], neighbours: int
) -> dict[str, float]:
    """Return how far a tile's nearest tiles share its class, over every tile asked.

    Args:
        distances: The distance between every pair of tiles. (T, T)
        labels: The class each tile carries, in the same order.
        neighbours: How many nearest tiles a precision and a recall are counted over.

    Returns:
        metrics: The precision, recall and F1 at that many neighbours, and the mean
            average precision over the whole ranking, averaged over the tiles.
    """
    held = np.asarray(labels)
    itself = np.eye(len(held), dtype=bool)
    # A tile is never its own neighbour, so its own column is put out of reach.
    ranked = np.argsort(np.where(itself, np.inf, distances), axis=1)[:, :-1]  # (T, T-1)
    relevant = held[ranked] == held[:, None]  # (T, T-1)
    total = np.maximum(relevant.sum(axis=1), 1)  # (T,)
    taken = relevant[:, :neighbours]  # (T, k)
    precision = taken.mean(axis=1)  # (T,)
    recall = taken.sum(axis=1) / total  # (T,)
    together = np.maximum(precision + recall, np.finfo(float).eps)
    # Average precision reads the whole ranking, not the neighbours alone.
    hits = np.cumsum(relevant, axis=1)  # (T, T-1)
    ranks = np.arange(1, relevant.shape[1] + 1)  # (T-1,)
    return {
        f"precision@{neighbours}": float(precision.mean()),
        f"recall@{neighbours}": float(recall.mean()),
        f"f1@{neighbours}": float((2 * precision * recall / together).mean()),
        "map": float(((relevant * hits / ranks).sum(axis=1) / total).mean()),
    }


def silhouette_by_class(
    distances: np.ndarray, labels: Sequence[str]
) -> dict[str, float]:
    """Return how tightly each class sits together against the nearest other class.

    Args:
        distances: The distance between every pair of tiles. (T, T)
        labels: The class each tile carries, in the same order.

    Returns:
        silhouettes: The silhouette over every tile, and the mean over each class.
    """
    held = np.asarray(labels)
    samples = silhouette_samples(distances, held, metric="precomputed")  # (T,)
    return {"silhouette": float(samples.mean())} | {
        f"silhouette/{name}": float(samples[held == name].mean())
        for name in sorted(set(labels))
    }


def class_distances(
    distances: np.ndarray, labels: Sequence[str]
) -> tuple[list[str], np.ndarray]:
    """Return how far each class stands from each, averaged over the tiles they hold.

    Args:
        distances: The distance between every pair of tiles. (T, T)
        labels: The class each tile carries, in the same order.

    Returns:
        classes: The classes, in the order the matrix holds them.
        matrix: The mean distance between the tiles of two classes. (C, C)
    """
    held = np.asarray(labels)
    classes = sorted(set(labels))
    # A tile against itself says nothing, so it is left out of every mean.
    counted = np.where(np.eye(len(held), dtype=bool), np.nan, distances)
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


def class_separation(matrix: np.ndarray) -> dict[str, float]:
    """Return what the class distances come to, whichever classes they were measured on.

    Args:
        matrix: The mean distance between the tiles of two classes. (C, C)

    Returns:
        separation: How far a class sits from itself, how far from another, and the
            gap between the two, which is what a latent space is asked for.
    """
    within = float(np.mean(np.diag(matrix)))
    between = float(np.mean(matrix[~np.eye(len(matrix), dtype=bool)]))
    return {"within": within, "between": between, "separation": between - within}
