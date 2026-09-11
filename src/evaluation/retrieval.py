"""The retrieval metrics the paper reads a latent space with, a class its relevance."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def retrieval_metrics(
    latents: Tensor, classes: Sequence[str], neighbours: int
) -> dict[str, float]:
    """Return how much of what each latent retrieves shares its own class.

    Args:
        latents: One vector per feature, of unit length. (N, D)
        classes: The class of each of them, in that same order.
        neighbours: How many nearest latents one query reads, or every other
            latent where the split holds fewer.

    Returns:
        metrics: Over those neighbours the precision, the recall against every
            feature of the query's class, their F1, and the average precision
            under "map". Each is averaged over the queries whose class holds
            another feature of the split, a query whose class holds none being
            one no ranking can answer.
    """
    names = sorted(set(classes))
    labels = torch.tensor([names.index(one) for one in classes])  # (N,)
    similarity = latents @ latents.T  # (N, N)
    similarity.fill_diagonal_(float("-inf"))
    read = min(neighbours, len(classes) - 1)
    found = labels[similarity.topk(read, dim=1).indices]  # (N, k)
    relevant = (found == labels.unsqueeze(1)).to(latents.dtype)  # (N, k)
    held = torch.bincount(labels)[labels] - 1  # (N,)
    counted = held > 0  # (N,)
    ranks = torch.arange(1, read + 1, dtype=latents.dtype)  # (k,)
    precision = relevant.mean(dim=1)  # (N,)
    recall = relevant.sum(dim=1) / held.clamp(min=1)  # (N,)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)  # (N,)
    reachable = torch.minimum(held, torch.tensor(read)).clamp(min=1)  # (N,)
    average = (relevant.cumsum(dim=1) / ranks * relevant).sum(dim=1) / reachable  # (N,)
    return {
        "precision": float(precision[counted].mean()),
        "recall": float(recall[counted].mean()),
        "f1": float(f1[counted].mean()),
        "map": float(average[counted].mean()),
        "queries": int(counted.sum()),
    }
