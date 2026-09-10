"""How well every feature retrieves features of its own class."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor


def neighbours(embeddings: Tensor, count: int) -> Tensor:
    """Return the nearest other features of each, by cosine similarity.

    Args:
        embeddings: One unit vector per feature. (N, L)
        count: How many neighbours to return, or every other feature where
            there are fewer.

    Returns:
        nearest: The indices of the nearest features of each, nearest first,
            the feature itself never among them. (N, min(count, N - 1))
    """
    similarity = embeddings @ embeddings.T  # (N, N)
    similarity.fill_diagonal_(-math.inf)
    return similarity.topk(min(count, embeddings.shape[0] - 1), dim=1).indices


def retrieval_metrics(
    embeddings: Tensor, classes: Sequence[str], count: int
) -> dict[str, float]:
    """Return how well every feature retrieves features of its own class.

    Args:
        embeddings: One unit vector per feature. (N, L)
        classes: The class of each feature, in the same order.
        count: How many nearest features count.

    Returns:
        metrics: Under "precision_at_k" the share of the nearest features that
            share the class, under "mean_average_precision" the precision
            averaged at each of the nearest that does, and under
            "nearest_neighbour_accuracy" the share of features whose nearest
            shares their class, all averaged over every feature as a query.
            Not a number where the split holds fewer than two features.
    """
    names = sorted(set(classes))
    labels = torch.tensor([names.index(one) for one in classes])  # (N,)
    nearest = neighbours(embeddings.cpu(), count)  # (N, k)
    if nearest.shape[1] == 0:
        return {
            "precision_at_k": math.nan,
            "mean_average_precision": math.nan,
            "nearest_neighbour_accuracy": math.nan,
        }
    hits = (labels[nearest] == labels.unsqueeze(1)).float()  # (N, k)
    ranks = torch.arange(1, hits.shape[1] + 1)  # (k,)
    precision = hits.cumsum(dim=1) / ranks  # (N, k)
    average = (precision * hits).sum(dim=1) / hits.sum(dim=1).clamp(min=1)  # (N,)
    return {
        "precision_at_k": float(hits.mean()),
        "mean_average_precision": float(average.mean()),
        "nearest_neighbour_accuracy": float(hits[:, 0].mean()),
    }
