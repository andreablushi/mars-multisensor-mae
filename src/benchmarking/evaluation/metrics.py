"""How well a head's points fit SWIM's, target by target."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

import numpy as np

from dataset.loader import TARGETS


def rmse_and_r2(
    predictions: Mapping[str, Mapping[str, np.ndarray]],
    targets: Mapping[str, np.ndarray],
) -> dict[str, dict[str, dict[str, float]]]:
    """Return each target's RMSE and R2 over every point of the tiles.

    Args:
        predictions: Every point's prediction of each target over some tiles, keyed
            by protocol, then by tile. (P * P, Z)
        targets: SWIM's values there, keyed by tile over at least those tiles.

    Returns:
        scores: The RMSE and R2, keyed by protocol, then by target and metric.
    """
    scores = defaultdict(dict)
    for protocol, by_tile in predictions.items():
        tiles = sorted(by_tile)
        predicted, wanted = (
            np.stack([held[tile] for tile in tiles]) for held in (by_tile, targets)
        )  # (T, P * P, Z)
        for at, name in enumerate(TARGETS):
            score, truth = predicted[..., at], wanted[..., at]
            error = np.sum((score - truth) ** 2)
            scores[protocol][name] = {
                "RMSE": float(np.sqrt(error / truth.size)),
                "R2": float(1 - error / np.sum((truth - truth.mean()) ** 2)),
            }
    return dict(scores)
