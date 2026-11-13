"""What the heads are trained to make small."""

from __future__ import annotations

from torch import Tensor


def mse(prediction: Tensor, target: Tensor, spread: Tensor, scored: Tensor) -> Tensor:
    """Return the mean squared error over the scored points, each target in its spread.

    Args:
        prediction: Every point's prediction of each target. (B, P * P, Z)
        target: SWIM's value there. (B, P * P, Z)
        spread: Each target's standard deviation over the training tiles. (Z,)
        scored: Which points are scored. (B, P * P)

    Returns:
        error: The squared error in spreads, averaged over the scored points and the
            targets. ()
    """
    return (((prediction - target) / spread) ** 2)[scored].mean()
