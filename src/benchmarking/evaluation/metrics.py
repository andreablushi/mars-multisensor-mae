"""What a head's consistency comes to against SWIM's ice, depth by depth."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

DEPTHS = ("1_5m", "5m")

COMBOS = ("neither", "1-5 m only", "below 5 m only", "both")


def ice_combos(icy: np.ndarray) -> np.ndarray:
    """Return where ice is found at each depth as one of COMBOS, by its index.

    Args:
        icy: Whether each depth is icy. (..., Z)

    Returns:
        combos: The index in COMBOS of the depths icy. (...)
    """
    return icy.astype(int) @ np.array([1, 2])


def ice_scores(
    consistency: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    icy: Mapping[str, np.ndarray],
    segmented: bool,
) -> dict[str, dict[str, float]]:
    """Return how well each depth's consistency ranks SWIM's ice and fits SWIM's own.

    Args:
        consistency: The consistency at each depth of some tiles, keyed by the
            tile. (Z,) or (P * P, Z)
        targets: SWIM's consistency there, keyed alike over at least those tiles.
        icy: Whether SWIM finds ice there, keyed alike over at least those tiles.
        segmented: Whether every point is scored, rather than the tile. Only the
            tiles SWIM finds both icy and dry at a depth are then scored: AUROC and
            AUPRC as the mean of what each tile's points come to, RMSE over their
            points, and R2 around each tile's own mean.

    Returns:
        scores: The AUROC, AUPRC, RMSE and R2, keyed by depth, then by metric.
    """

    def ranked(truth: np.ndarray, score: np.ndarray) -> dict[str, float]:
        """Return how well the score ranks the icy above the dry."""
        return {
            "AUROC": float(roc_auc_score(truth, score)),
            "AUPRC": float(average_precision_score(truth, score)),
        }

    tiles = sorted(consistency)
    predicted, wanted, truth = (
        np.stack([held[tile] for tile in tiles]) for held in (consistency, targets, icy)
    )  # (T, Z) or (T, P * P, Z)
    scores = {}
    for at, depth in enumerate(DEPTHS):
        score, target, held = predicted[..., at], wanted[..., at], truth[..., at]
        if segmented:
            mixed = (0 < held.mean(1)) & (held.mean(1) < 1)
            score, target, held = score[mixed], target[mixed], held[mixed]
            each = [ranked(*one) for one in zip(held, score, strict=True)]
            centre = target.mean(1, keepdims=True)
            scores[depth] = {
                metric: float(np.mean([one[metric] for one in each]))
                for metric in each[0]
            }
        else:
            centre = target.mean()
            scores[depth] = ranked(held, score)
        error = np.sum((score - target) ** 2)
        scores[depth] |= {
            "RMSE": float(np.sqrt(error / target.size)),
            "R2": float(1 - error / np.sum((target - centre) ** 2)),
        }
    return scores


def ice_confusion(
    consistency: Mapping[str, np.ndarray],
    icy: Mapping[str, np.ndarray],
    thresholds: np.ndarray,
) -> np.ndarray:
    """Return how many tiles of each SWIM combo are predicted as each combo.

    Args:
        consistency: The consistency at each depth of some tiles, keyed by the
            tile. (Z,)
        icy: Whether SWIM finds ice there, keyed alike over at least those tiles.
        thresholds: SWIM's consistency from which each depth is icy. (Z)

    Returns:
        counts: The tiles of each SWIM combo by row and predicted combo by column,
            both in COMBOS order. (len(COMBOS), len(COMBOS))
    """
    tiles = sorted(consistency)
    truth = ice_combos(np.stack([icy[tile] for tile in tiles]))
    predicted = ice_combos(
        np.stack([consistency[tile] for tile in tiles]) >= thresholds
    )
    return confusion_matrix(truth, predicted, labels=range(len(COMBOS)))
