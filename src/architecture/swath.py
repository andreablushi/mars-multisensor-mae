"""The latent space, where a feature is the average of the tokens it was read as."""

from __future__ import annotations

from torch import Tensor


def feature_swath(tokens: Tensor, counted: Tensor) -> Tensor:
    """Return the swath of each feature, the average of the tokens that count.

    Args:
        tokens: The tokens as the cross-sensor encoder hands them, of one
            instrument or of every one side by side. (B, K, D)
        counted: Which of them count. (B, K)

    Returns:
        swath: Their global average, zero where none counts. (B, D)
    """
    weight = counted.unsqueeze(-1).to(tokens.dtype)  # (B, K, 1)
    return (tokens * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1)  # (B, D)
