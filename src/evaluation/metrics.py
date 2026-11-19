"""What the similarities between tiles come to against their classes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    silhouette_samples,
)

from evaluation.models.metrics import Metrics


def metrics_by_view(
    vectors: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    raw: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    classes: Mapping[str, str],
    k: int,
) -> dict[tuple[str, str], Metrics]:
    """Return the metrics of every pooling and view.

    Args:
        vectors: Each tile's vector, keyed by pooling, then view, then tile.
        raw: The training tiles' vectors before normalisation, which RankMe reads,
            keyed alike.
        classes: The class each tile earned, keyed by the tile.
        k: How many neighbours the kNN vote reads.

    Returns:
        metrics: Each view's metrics, keyed by its pooling and view.
    """
    metrics = {}
    for pooling, by_view in vectors.items():
        tiles = sorted(next(iter(by_view.values())))
        labels = [classes[tile] for tile in tiles]
        classed = sorted(set(labels))
        stacked = {
            view: np.array([by_tile[tile] for tile in tiles], np.float64)
            for view, by_tile in by_view.items()
        }
        cross = cross_sensor_metrics(
            {view: held for view, held in stacked.items() if view != "all"}
        )
        for view, held in stacked.items():
            similarities = held @ held.T
            ranked = ranked_neighbours(similarities)
            metrics[pooling, view] = Metrics(
                retrieval=retrieval_metrics(ranked, labels),
                knn=knn_metrics(ranked, labels, k),
                classes=classed,
                distances=class_distance_metrics(similarities, labels, classed),
                silhouette=silhouette_scores(similarities, labels),
                cross=cross[view],
                rank=rank_metrics(
                    np.array(list(raw[pooling][view].values()), np.float64)
                ),
            )
    return metrics


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
    ranked: np.ndarray,
    labels: Sequence[str],
    hidden: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return how early each tile's ranking reaches the tiles sharing its class.

    Args:
        ranked: Each tile's other tiles, most similar first. (T, T-1)
        labels: The class each tile carries, in the same order.
        hidden: Which tiles each tile's ranking leaves out, or None for none. (T, T)

    Returns:
        scores: Every tile's average precision over the whole ranking. (T,)
    """
    held = np.asarray(labels)
    kept = (
        np.ones(ranked.shape, bool)
        if hidden is None
        else ~np.take_along_axis(hidden, ranked, axis=1)
    )  # (T, T-1)
    relevant = (held[ranked] == held[:, None]) & kept  # (T, T-1)
    total = relevant.sum(axis=1)  # (T,)
    found = np.cumsum(relevant, axis=1)  # (T, T-1)
    # The precision at the rank of every classmate, zero at every other rank
    gained = found / np.cumsum(kept, axis=1).clip(min=1) * relevant  # (T, T-1)
    return {"mAP": gained.sum(axis=1) / total}


def cross_sensor_metrics(
    vectors: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Return how early each instrument's vectors find the same tile in every other's.

    Args:
        vectors: Each instrument's tile vectors, in one order of tiles. (T, D)

    Returns:
        scores: By view, then "cross-sensor mAP from <query>": the mean of one over
            the rank of the tile's own vector among the view's, none under "all".
    """
    scores = {view: {} for view in (*vectors, "all")}
    for query, asked in vectors.items():
        metric = f"cross-sensor mAP from {query}"
        for gallery, held in vectors.items():
            if gallery == query:
                continue
            similarities = asked @ held.T  # (T, T)
            ahead = (similarities > np.diag(similarities)[:, None]).sum(axis=1)
            scores[gallery][metric] = float((1 / (1 + ahead)).mean())
    return scores


def modality_gap_by_pooling(
    vectors: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
) -> dict[tuple[str, str], float]:
    """Return the RMG of every pair of instruments (Schrodi et al. 2025, Eq. 1).

    Args:
        vectors: Each tile's unit vector, keyed by pooling, then view, then tile.

    Returns:
        gaps: RMG from 0 to 1, keyed by pooling and "<one>-<other>"; none for "all".
    """
    gaps = {}
    for pooling, by_view in vectors.items():
        tiles = sorted(next(iter(by_view.values())))
        total = len(tiles)
        stacked = {
            view: np.array([by_tile[tile] for tile in tiles], np.float64)
            for view, by_tile in by_view.items()
            if view != "all"
        }
        spread = {
            view: (1 - (held.sum(0) @ held.sum(0) - total) / (total * (total - 1))) / 2
            for view, held in stacked.items()
        }
        for one, other in combinations(stacked, 2):
            cosines = (stacked[one] * stacked[other]).sum(1)  # (T,)
            matched = max(float(np.mean((1 - cosines) / 2)), 1e-3)
            gaps[pooling, f"{one}-{other}"] = matched / (
                (spread[one] + spread[other]) / 2 + matched
            )
    return gaps


def knn_metrics(ranked: np.ndarray, labels: Sequence[str], k: int) -> dict[str, float]:
    """Return how well the class each tile's k nearest tiles vote for matches its own.

    Args:
        ranked: Each tile's other tiles, most similar first. (T, T-1)
        labels: The class each tile carries, in the same order.
        k: How many neighbours vote, a tie going to the nearest.

    Returns:
        scores: Accuracy and balanced accuracy.
    """
    votes = np.asarray(labels)[ranked[:, :k]]  # (T, k)
    predicted = []
    for row in votes:
        counts = Counter(row)
        most = max(counts.values())
        predicted.append(next(one for one in row if counts[one] == most))
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced accuracy": float(balanced_accuracy_score(labels, predicted)),
    }


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


def silhouette_scores(similarities: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    """Return how much nearer each tile sits to its own class than to the next one.

    Args:
        similarities: The cosine of every pair of tiles. (T, T)
        labels: The class each tile carries, in the same order.

    Returns:
        scores: Every tile's silhouette over cosine distances, from -1 to 1. (T,)
    """
    # Rounding leaves the diagonal and near duplicates a hair off zero
    distances = np.clip(1 - similarities, 0, None)
    np.fill_diagonal(distances, 0)
    return silhouette_samples(distances, labels, metric="precomputed")


def rank_metrics(vectors: np.ndarray) -> dict[str, float]:
    """Return the RankMe effective rank of some vectors (Garrido et al. 2023, Eq. 1).

    Args:
        vectors: One vector per tile, not normalised. (T, D)

    Returns:
        scores: "RankMe", the exponential of the entropy of the singular values summing
            to one, and "RankMe max", min(T, D), the most it reaches.
    """
    singular = np.linalg.svd(vectors, compute_uv=False)
    share = singular / singular.sum() + 1e-7
    return {
        "RankMe": float(np.exp(-np.sum(share * np.log(share)))),
        "RankMe max": float(min(vectors.shape)),
    }
