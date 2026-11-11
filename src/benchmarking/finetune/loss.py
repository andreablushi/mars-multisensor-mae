"""What the heads are trained to make small."""

from __future__ import annotations

from torch import Tensor


def mse(prediction: Tensor, target: Tensor, spread: Tensor) -> Tensor:
    """Return the mean squared error over every point and target, each in its spread.

    Args:
        prediction: Every point's prediction of each target. (B, P * P, Z)
        target: SWIM's value there. (B, P * P, Z)
        spread: Each target's standard deviation over the training tiles. (Z,)

    Returns:
        error: The squared error in spreads, averaged over the points and targets. ()
    """
    return (((prediction - target) / spread) ** 2).mean()
