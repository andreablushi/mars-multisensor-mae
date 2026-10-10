"""What the distances between tiles come to against their classes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import silhouette_samples
from umap import UMAP

TOP_K = (1, 5, 10, 20)


def retrieval_by_tile(
    distances: np.ndarray, labels: Sequence[str]
) -> dict[str, np.ndarray]:
    """Return how far each tile's nearest tiles share its class.

    Args:
        distances: The distance between every pair of tiles. (T, T)
        labels: The class each tile carries, in the same order.

    Returns:
        scores: Every tile's precision, recall and F1 at every k of TOP_K. (T,)
    """
    held = np.asarray(labels)
    itself = np.eye(len(held), dtype=bool)
    # A tile is never its own neighbour, so its own column is put out of reach.
    ranked = np.argsort(np.where(itself, np.inf, distances), axis=1)[:, :-1]  # (T, T-1)
    relevant = held[ranked] == held[:, None]  # (T, T-1)
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


def umap_projection(distances: np.ndarray, seed: int) -> np.ndarray:
    """Return every tile laid on a plane by UMAP, read off the tile distances.

    Args:
        distances: The distance between every pair of tiles. (T, T)
        seed: What the layout is drawn with, so the same distances lay out the same.

    Returns:
        projection: Where each tile lands on the plane, in the same order. (T, 2)
    """
    return UMAP(metric="precomputed", random_state=seed).fit_transform(distances)
