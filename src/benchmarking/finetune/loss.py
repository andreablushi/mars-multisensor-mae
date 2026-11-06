"""What a head minimises: the cross-entropy of its ice against SWIM's."""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional


def ice_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Return the binary cross-entropy of a head's ice, over every depth and tile.

    Args:
        logits: The head's logits. (B, Z) or (B, P * P, Z)
        targets: Whether SWIM finds ice there, as 0 or 1, shaped alike.

    Returns:
        loss: Its mean over every depth, tile and point.
    """
    return functional.binary_cross_entropy_with_logits(logits, targets)
