"""How far tiles stand from each other, and what that comes to against their classes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from sklearn.metrics import silhouette_samples
from torch import Tensor
from umap import UMAP

from evaluation.evaluate import TileTokens

TOP_K = (1, 5, 10, 20)
PAIR_BATCH = 4


def chamfer_distances(
    tiles: Sequence[TileTokens], minimal_chamfer_cell_distance: int | None = None
) -> Tensor:
    """Return how far every tile stands from every other, over the tokens they hold.

    A tile is its weighted tokens and nothing else, so two are compared by matching
    each token of one to the nearest token of the other and averaging both ways by
    weight. The cost of a match is the cosine distance between two tokens, which
    their unit length puts in [0, 1], so the distance is already normalised. A token
    whose neighbourhood holds nothing to match stands a whole mismatch from that tile.

    Args:
        tiles: One set of tokens per tile, in the order the distances are wanted.
        minimal_chamfer_cell_distance: How far, in cells along any axis, a
            token may be matched from its own cell, or None to match it anywhere
            in the other tile.

    Returns:
        distances: The distance between every pair of tiles, in [0, 1]. (T, T)
    """
    counts = torch.tensor([int((one.weight > 0).sum()) for one in tiles])  # (T,)
    width = int(counts.max())
    values = tiles[0].values.new_zeros(len(tiles), width, tiles[0].values.shape[-1])
    offsets = values.new_zeros(len(tiles), width, 3)
    weights = values.new_zeros(len(tiles), width)
    for at, tile in enumerate(tiles):
        compared = tile.weight > 0
        values[at, : counts[at]] = tile.values[compared]
        offsets[at, : counts[at]] = tile.offset[compared].to(values.dtype)
        weights[at, : counts[at]] = tile.weight[compared]
    held = weights > 0  # (T, W)
    distances = values.new_zeros(len(tiles), len(tiles))
    # Both ways round are the same sum, so only a tile against those after it is read.
    for at in range(len(tiles)):
        for start in range(at, len(tiles), PAIR_BATCH):
            rest = slice(start, min(start + PAIR_BATCH, len(tiles)))
            cost = (
                1.0 - torch.einsum("qd,tpd->tqp", values[at], values[rest])
            ) / 2  # (R, W, W)
            matched = held[at].view(1, -1, 1) & held[rest].unsqueeze(1)
            if minimal_chamfer_cell_distance is not None:
                for axis in range(3):
                    apart = (
                        offsets[at, :, axis][None, :, None]
                        - offsets[rest, :, axis][:, None]
                    ).abs()
                    matched &= apart <= minimal_chamfer_cell_distance
            cost.masked_fill_(~matched, torch.inf)
            forward = cost.amin(dim=2).nan_to_num(posinf=1.0)
            backward = cost.amin(dim=1).nan_to_num(posinf=1.0)
            distances[at, rest] = (
                (forward * weights[at]).sum(-1) + (backward * weights[rest]).sum(-1)
            ) / 2
    return (distances + distances.T).fill_diagonal_(0.0)


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
