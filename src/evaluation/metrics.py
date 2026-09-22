"""How far tiles stand from each other, and what that comes to against their classes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from sklearn.metrics import silhouette_samples
from torch import Tensor
from umap import UMAP

from architecture.models import TileGrid

TOP_K = (1, 5, 10, 20)


def chamfer_distances(
    grids: Sequence[TileGrid], minimal_chamfer_cell_distance: int | None = None
) -> Tensor:
    """Return how far every tile stands from every other, over the cells they hold.

    A tile is its occupied cells and nothing else, so two are compared by matching
    each cell of one to the nearest cell of the other and averaging both ways. The
    cost of a match is the cosine distance between two cells, which the fusion's
    unit vectors put in [0, 1], so the distance is already normalised. A cell whose
    neighbourhood holds nothing to match stands a whole mismatch from that tile.

    Args:
        grids: One grid per tile, in the order the distances are wanted.
        minimal_chamfer_cell_distance: How far, in cells along either axis, a
            cell may be matched from its own offset, or None to match it anywhere
            in the other tile.

    Returns:
        distances: The distance between every pair of tiles, in [0, 1]. (T, T)

    Raises:
        ValueError: When a tile holds no occupied cell to be measured over.
    """
    counts = torch.tensor([int(one.occupied.sum()) for one in grids])  # (T,)
    if not counts.all():
        raise ValueError("a tile holding no occupied cell cannot be measured")
    device = grids[0].values.device
    width = int(counts.max())
    values = grids[0].values.new_zeros(len(grids), width, grids[0].values.shape[-1])
    offsets = values.new_zeros(len(grids), width, 2)
    for at, grid in enumerate(grids):
        values[at, : counts[at]] = grid.values[grid.occupied]
        offsets[at, : counts[at]] = grid.offset[grid.occupied].to(values.dtype)
    counted = counts.to(device, values.dtype)  # (T,)
    held = torch.arange(width, device=device) < counted.unsqueeze(1)  # (T, W)
    distances = values.new_zeros(len(grids), len(grids))
    # Both ways round are the same sum, so only a tile against those after it is read.
    for at in range(len(grids)):
        rest = slice(at, len(grids))
        cost = (
            1.0 - torch.einsum("qd,tpd->tqp", values[at], values[rest])
        ) / 2  # (R, W, W)
        matched = held[at].view(1, -1, 1) & held[rest].unsqueeze(1)  # (R, W, W)
        if minimal_chamfer_cell_distance is not None:
            # Taken an axis at a time, so no difference is held for both at once.
            here, there = offsets[at].unbind(-1), offsets[rest].unbind(-1)
            east = (here[0].view(1, -1, 1) - there[0].unsqueeze(1)).abs()
            north = (here[1].view(1, -1, 1) - there[1].unsqueeze(1)).abs()
            matched &= (
                torch.maximum(east, north) <= minimal_chamfer_cell_distance
            )  # (R, W, W)
        cost.masked_fill_(matched.logical_not_(), torch.inf)
        forward = cost.amin(dim=2).nan_to_num(posinf=1.0)  # (R, W)
        backward = cost.amin(dim=1).nan_to_num(posinf=1.0)  # (R, W)
        distances[at, rest] = (
            (forward * held[at]).sum(-1) / counted[at]
            + (backward * held[rest]).sum(-1) / counted[rest]
        ) / 2
    return (distances + distances.T).fill_diagonal_(0.0)


def retrieval_metrics(distances: np.ndarray, labels: Sequence[str]) -> dict[str, float]:
    """Return how far a tile's nearest tiles share its class, over every tile asked.

    Args:
        distances: The distance between every pair of tiles. (T, T)
        labels: The class each tile carries, in the same order.

    Returns:
        metrics: The precision, recall and F1 at every k of TOP_K, averaged over
            the tiles.
    """
    held = np.asarray(labels)
    itself = np.eye(len(held), dtype=bool)
    # A tile is never its own neighbour, so its own column is put out of reach.
    ranked = np.argsort(np.where(itself, np.inf, distances), axis=1)[:, :-1]  # (T, T-1)
    relevant = held[ranked] == held[:, None]  # (T, T-1)
    total = np.maximum(relevant.sum(axis=1), 1)  # (T,)
    metrics = {}
    for k in TOP_K:
        taken = relevant[:, :k]  # (T, k)
        precision = taken.mean(axis=1)  # (T,)
        recall = taken.sum(axis=1) / total  # (T,)
        together = np.maximum(precision + recall, np.finfo(float).eps)
        metrics |= {
            f"precision@{k}": float(precision.mean()),
            f"recall@{k}": float(recall.mean()),
            f"f1@{k}": float((2 * precision * recall / together).mean()),
        }
    return metrics


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


def umap_projection(distances: np.ndarray, seed: int) -> np.ndarray:
    """Return every tile laid on a plane by UMAP, read off the tile distances.

    Args:
        distances: The distance between every pair of tiles. (T, T)
        seed: What the layout is drawn with, so the same distances lay out the same.

    Returns:
        projection: Where each tile lands on the plane, in the same order. (T, 2)
    """
    return UMAP(metric="precomputed", random_state=seed).fit_transform(distances)
