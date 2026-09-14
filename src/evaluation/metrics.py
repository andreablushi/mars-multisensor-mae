"""Every number one evaluation measures, of a latent space and of a reconstruction."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from sklearn.metrics import silhouette_samples
from torch import Tensor

from training.loss import normalised_patches


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


def reconstruction_metrics(
    prediction: Tensor, values: Tensor, valid: Tensor, weight: Tensor
) -> dict[str, float]:
    """Return how closely one instrument's predicted patches stand to the true ones.

    Args:
        prediction: The predicted patches. (B, K, *P)
        values: The true ones, as the model was handed them. (B, K, *P)
        valid: Whether each sample is a measurement, broadcastable to them. (B, K, *P')
        weight: How much each patch counts. (B, K)

    Returns:
        metrics: Under "mse" the squared error over the measured samples, under
            "r2" the share of a patch's variance it accounts for, the target
            carrying unit variance so the error is already what is left, and
            under "psnr" the peak signal to noise ratio in decibels, read
            against the range of the patch's own measured samples. Each is
            averaged over the counted patches, and zero where none counts.
    """
    target, counted = normalised_patches(values, valid)  # (B, K, *P)
    over = tuple(range(2, values.dim()))
    samples = counted.sum(dim=over).clamp(min=1)  # (B, K)
    error = ((prediction - target) ** 2 * counted).sum(dim=over) / samples  # (B, K)
    measured = counted > 0  # (B, K, *P)
    peak = target.masked_fill(~measured, -torch.inf).amax(dim=over)  # (B, K)
    trough = target.masked_fill(~measured, torch.inf).amin(dim=over)  # (B, K)
    spread = (peak - trough).clamp(min=1e-6)  # (B, K)
    ratio = 10 * (spread**2 / error.clamp(min=1e-12)).log10()  # (B, K)
    weight = weight.to(values.dtype)  # (B, K)
    held = weight.sum().clamp(min=1)  # ()
    return {
        "mse": float((error * weight).sum() / held),
        "r2": float(((1 - error) * weight).sum() / held),
        "psnr": float((ratio * weight).sum() / held),
    }
