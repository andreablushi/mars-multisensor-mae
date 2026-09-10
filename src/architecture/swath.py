"""The latent space, where every token is a direction and a feature is its swath."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional


def unit_directions(tokens: Tensor, kappa: float, training: bool) -> Tensor:
    """Return every token as a unit vector, drawn about its mean while training.

    Args:
        tokens: The tokens as the cross-sensor encoder hands them. (B, K, L)
        kappa: How tightly a draw stays about its mean direction.
        training: Whether to draw about the mean rather than hand it back.

    Returns:
        directions: Unit vectors, the mean direction of each token when
            evaluating and a draw about it when training. (B, K, L)
    """
    mean = functional.normalize(tokens, dim=-1)  # (B, K, L)
    if not training:
        return mean
    drawn = mean + torch.randn_like(mean) / math.sqrt(kappa)  # (B, K, L)
    return functional.normalize(drawn, dim=-1)  # (B, K, L)


def feature_swath(directions: Tensor, present: Tensor) -> Tensor:
    """Return the swath of each feature, where its tokens point on average.

    Args:
        directions: Unit vectors, every instrument's tokens side by side.
            (B, K, L)
        present: Which of them count. (B, K)

    Returns:
        swath: The unit vector along their sum, zero where none counts. (B, L)
    """
    weight = present.unsqueeze(-1).to(directions.dtype)  # (B, K, 1)
    summed = (directions * weight).sum(dim=1)  # (B, L)
    return functional.normalize(summed, dim=-1)  # (B, L)
