"""Bringing a feature's instruments together into the one vector that stands for it."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional


def instrument_vector(tokens: Tensor, counted: Tensor) -> Tensor:
    """Return what one instrument makes of each feature, its tokens averaged.

    Args:
        tokens: The instrument's tokens as the cross-sensor encoder hands them.
            (B, K, D)
        counted: Which of them count, its visible tokens while training and
            every present one when embedding. (B, K)

    Returns:
        vector: Their global average, zero where none counts. (B, D)
    """
    weight = counted.unsqueeze(-1).to(tokens.dtype)  # (B, K, 1)
    return (tokens * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1)  # (B, D)


def feature_latent(tokens: dict[str, Tensor], counted: dict[str, Tensor]) -> Tensor:
    """Return the one vector standing for each feature, over every instrument it holds.

    Each instrument is averaged on its own and set to unit length before they
    are added, so an instrument counts as much as any other however many
    patches it was drawn with, and one the feature does not hold counts for
    nothing. What makes the sum mean anything is the contrastive term, which
    trains the instruments of a feature to point the same way.

    Args:
        tokens: Each instrument's tokens as the cross-sensor encoder hands
            them, keyed as ODE names it. (B, K, D)
        counted: Which of each instrument's tokens count. (B, K)

    Returns:
        latent: The unit vector along their sum, zero for a feature holding no
            counted token of any instrument. (B, D)
    """
    vectors, held = [], []
    for name, one in tokens.items():
        vectors.append(
            functional.normalize(instrument_vector(one, counted[name]), dim=-1)
        )  # (B, D)
        held.append(counted[name].any(dim=1))  # (B,)
    stacked = torch.stack(vectors, dim=1)  # (B, J, D)
    weight = torch.stack(held, dim=1).unsqueeze(-1).to(stacked.dtype)  # (B, J, 1)
    return functional.normalize((stacked * weight).sum(dim=1), dim=-1)  # (B, D)
