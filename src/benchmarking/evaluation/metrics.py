"""How well a head's consistency fits SWIM's own, depth by depth."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

DEPTHS = ("1_5m", "5m")


def ice_scores(
    consistency: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    mapped: bool,
) -> dict[str, dict[str, float]]:
    """Return how well each depth's consistency fits SWIM's.

    Args:
        consistency: The consistency at each depth of some tiles, keyed by the
            tile. (Z,) or (P * P, Z)
        targets: SWIM's consistency there, keyed alike over at least those tiles.
        mapped: Whether every point is scored, rather than the tile. Only the
            tiles whose consistency SWIM finds varying at a depth are then scored,
            each point as its departure from its own tile's mean.

    Returns:
        scores: The RMSE and R2, keyed by depth, then by metric.
    """
    tiles = sorted(consistency)
    predicted, wanted = (
        np.stack([held[tile] for tile in tiles]) for held in (consistency, targets)
    )  # (T, Z) or (T, P * P, Z)
    scores = {}
    for at, depth in enumerate(DEPTHS):
        score, target = predicted[..., at], wanted[..., at]
        if mapped:
            varied = target.std(1) > 0
            score, target = (
                one[varied] - one[varied].mean(1, keepdims=True)
                for one in (score, target)
            )
        error = np.sum((score - target) ** 2)
        scores[depth] = {
            "RMSE": float(np.sqrt(error / target.size)),
            "R2": float(1 - error / np.sum((target - target.mean()) ** 2)),
        }
    return scores
