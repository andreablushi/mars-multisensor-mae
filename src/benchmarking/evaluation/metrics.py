"""What a head's chance of ice comes to against SWIM's, depth by depth."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from sklearn.metrics import average_precision_score, jaccard_score, roc_auc_score

DEPTHS = ("1_5m", "5m")

THRESHOLD = 0.5


def ice_scores(
    probabilities: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    segmented: bool,
) -> dict[str, dict[str, float]]:
    """Return each depth's AUROC and AUPRC, and its mIoU when every point is scored.

    Args:
        probabilities: The chance of ice at each depth of some tiles, keyed by the
            tile. (Z,) or (P * P, Z)
        targets: Whether SWIM finds ice there, as 0 or 1, keyed alike over at least
            those tiles.
        segmented: Whether every point is scored, rather than the tile.

    Returns:
        scores: Each metric, keyed by depth, then by metric.
    """
    tiles = sorted(probabilities)
    predicted, truth = (
        np.stack([held[tile] for tile in tiles]).reshape(-1, len(DEPTHS))
        for held in (probabilities, targets)
    )
    scores = {}
    for at, depth in enumerate(DEPTHS):
        chance, icy = predicted[:, at], truth[:, at]
        scores[depth] = {
            "AUROC": float(roc_auc_score(icy, chance)),
            "AUPRC": float(average_precision_score(icy, chance)),
        }
        if segmented:
            scores[depth]["mIoU"] = float(
                jaccard_score(icy, chance >= THRESHOLD, average="macro")
            )
    return scores
