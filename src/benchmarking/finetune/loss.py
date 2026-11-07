"""What the heads minimise: the squared error of their consistency against SWIM's."""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor


def ice_losses(
    consistency: Mapping[str, Tensor], targets: Tensor, covered: Tensor
) -> dict[str, Tensor]:
    """Return each head's mean squared error over the points its tokens cover.

    Args:
        consistency: Each head's consistency, keyed by task. (B, Z) or (B, P * P, Z)
        targets: SWIM's consistency at every point and depth. (B, P * P, Z)
        covered: Which grid points have a token in the cell around them. (B, P * P)

    Returns:
        losses: Each head's loss, keyed by task, the tile head's against the
            median over the covered points.
    """
    held = covered.any(1)  # (B)
    median = targets.masked_fill(~covered[..., None], float("nan")).nanmedian(1)
    tiles = (consistency["tile"] - median.values.nan_to_num()) ** 2
    points = (consistency["map"] - targets) ** 2 * covered[..., None]
    return {
        "tile": tiles[held].mean(),
        "map": points.sum() / (covered.sum() * targets.shape[-1]),
    }
