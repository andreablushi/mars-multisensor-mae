"""How far a latent space holds a class together, and one class from another."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import silhouette_samples
from torch import Tensor


def class_similarity(
    latents: Tensor, classes: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    """Return how alike the latents of each pair of classes are.

    Args:
        latents: One vector per feature, of unit length. (N, D)
        classes: The class of each of them, in that same order.

    Returns:
        similarity: The mean cosine similarity of every ordered pair of
            classes, a class against itself counting its own pairs alone and
            not a latent against itself. (C, C)
        names: The classes, in the order the matrix holds them.
    """
    names = sorted(set(classes))
    held = np.asarray(classes)
    measured = (latents @ latents.T).numpy()  # (N, N)
    similarity = np.full((len(names), len(names)), np.nan)  # (C, C)
    for row, one in enumerate(names):
        for column, other in enumerate(names):
            block = measured[np.ix_(held == one, held == other)]
            if row == column:
                block = block[~np.eye(len(block), dtype=bool)]
            if block.size:
                similarity[row, column] = block.mean()
    return similarity, names


def similarity_metrics(similarity: np.ndarray) -> dict[str, float]:
    """Return what those similarities come to, within a class and between two.

    Args:
        similarity: The mean cosine similarity of every ordered pair of
            classes. (C, C)

    Returns:
        metrics: Under "within" the mean similarity of a class to itself, under
            "between" the mean over every unordered pair of different classes,
            and under "separation" how far the first stands above the second.
            Each class weighs the same however many features it holds.
    """
    within = float(np.nanmean(np.diagonal(similarity)))
    between = float(np.nanmean(similarity[np.triu_indices(len(similarity), k=1)]))
    return {"within": within, "between": between, "separation": within - between}


def silhouette_metrics(latents: Tensor, classes: Sequence[str]) -> dict[str, float]:
    """Return how well each class stands apart from the rest, as a silhouette reads it.

    Args:
        latents: One vector per feature, of unit length. (N, D)
        classes: The class of each of them, in that same order.

    Returns:
        metrics: Under "silhouette" the mean over every feature of how much
            closer it sits to its own class than to the nearest other, from
            minus one to one, and under "silhouette/<class>" that mean over the
            features of one class.
    """
    held = np.asarray(classes)
    samples = silhouette_samples(latents.numpy(), held, metric="cosine")  # (N,)
    metrics = {"silhouette": float(samples.mean())}
    for one in sorted(set(classes)):
        metrics[f"silhouette/{one}"] = float(samples[held == one].mean())
    return metrics
