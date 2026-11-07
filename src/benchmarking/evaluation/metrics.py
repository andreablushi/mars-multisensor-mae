"""What a head's chance of ice comes to against SWIM's, depth by depth."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    jaccard_score,
    roc_auc_score,
)

DEPTHS = ("1_5m", "5m")

COMBOS = ("neither", "1-5 m only", "below 5 m only", "both")


def ice_combos(chance: np.ndarray, threshold: float) -> np.ndarray:
    """Return where ice is found at each depth as one of COMBOS, by its index.

    Args:
        chance: The chance of ice at each depth, or SWIM's 0 or 1. (..., Z)
        threshold: The chance at or above which a depth counts as icy.

    Returns:
        combos: The index in COMBOS of the depths counted icy. (...)
    """
    return (chance >= threshold) @ np.array([1, 2])


def ice_scores(
    probabilities: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    segmented: bool,
    threshold: float,
) -> dict[str, dict[str, float]]:
    """Return each depth's AUROC and AUPRC, within each mixed tile when segmented.

    Args:
        probabilities: The chance of ice at each depth of some tiles, keyed by the
            tile. (Z,) or (P * P, Z)
        targets: Whether SWIM finds ice there, as 0 or 1, keyed alike over at least
            those tiles.
        segmented: Whether every point is scored, rather than the tile. Each metric,
            mIoU added, is then the mean over the tiles SWIM finds both icy and dry
            at that depth, of what it comes to over the tile's points.
        threshold: The chance at or above which a point counts as icy, for mIoU.

    Returns:
        scores: Each metric, keyed by depth, then by metric.
    """

    def ranked(icy: np.ndarray, chance: np.ndarray) -> dict[str, float]:
        """Return how well the chance ranks the icy above the dry."""
        return {
            "AUROC": float(roc_auc_score(icy, chance)),
            "AUPRC": float(average_precision_score(icy, chance)),
        }

    tiles = sorted(probabilities)
    predicted, truth = (
        np.stack([held[tile] for tile in tiles]) for held in (probabilities, targets)
    )  # (T, Z) or (T, P * P, Z)
    scores = {}
    for at, depth in enumerate(DEPTHS):
        chance, icy = predicted[..., at], truth[..., at]
        if not segmented:
            scores[depth] = ranked(icy, chance)
            continue
        mixed = [
            ranked(held, one)
            | {"mIoU": float(jaccard_score(held, one >= threshold, average="macro"))}
            for one, held in zip(chance, icy, strict=True)
            if 0 < held.mean() < 1
        ]
        scores[depth] = {
            metric: float(np.mean([held[metric] for held in mixed]))
            for metric in mixed[0]
        }
    return scores


def ice_confusion(
    probabilities: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    threshold: float,
) -> np.ndarray:
    """Return how many tiles of each SWIM combo are predicted as each combo.

    Args:
        probabilities: The chance of ice at each depth of some tiles, keyed by the
            tile. (Z,)
        targets: Whether SWIM finds ice there, as 0 or 1, keyed alike over at least
            those tiles.
        threshold: The chance at or above which a depth counts as icy.

    Returns:
        counts: The tiles of each SWIM combo by row and predicted combo by column,
            both in COMBOS order. (len(COMBOS), len(COMBOS))
    """
    tiles = sorted(probabilities)
    truth, predicted = (
        ice_combos(np.stack([held[tile] for tile in tiles]), threshold)
        for held in (targets, probabilities)
    )
    return confusion_matrix(truth, predicted, labels=range(len(COMBOS)))
