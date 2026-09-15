"""Bringing what every sensor made of a feature into the one vector standing for it."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional


def instrument_vector(tokens: Tensor, counted: Tensor) -> Tensor:
    """Return what one instrument makes of each feature, its tokens averaged.

    Args:
        tokens: The sensor's tokens from the cross-sensor encoder. (B, K, D)
        counted: Which count: visible while training, present when embedding. (B, K)

    Returns:
        vector: Their global average, zero where none counts. (B, D)
    """
    # Convert valid token mask into broadcastable float weights
    weight = counted.unsqueeze(-1).to(tokens.dtype)  # (B, K, 1)
    # Compute masked spatial mean across patch tokens for a single instrument
    return (tokens * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1)  # (B, D)


def feature_latent(tokens: dict[str, Tensor], counted: dict[str, Tensor]) -> Tensor:
    """Return the one vector standing for each feature, over every instrument it holds.

    Each instrument is averaged on its own and set to unit length before they
    are added, so an instrument counts as much as any other however many
    patches it was drawn with, and one the feature does not hold counts for
    nothing. What makes the sum mean anything is the contrastive term, which
    trains the instruments of a feature to point the same way.

    Args:
        tokens: Each sensor's tokens from the cross-sensor encoder. (B, K, D)
        counted: Which of each instrument's tokens count. (B, K)

    Returns:
        latent: The unit vector along their sum, zero where none counted. (B, D)
    """
    vectors, held = [], []
    for name, one in tokens.items():
        # Mean-pool patch tokens per sensor and normalize to unit L2 length
        vectors.append(
            functional.normalize(instrument_vector(one, counted[name]), dim=-1)
        )  # (B, D)
        # Check whether the instrument has any valid tokens present in the batch sample
        held.append(counted[name].any(dim=1))  # (B,)
    # Stack per-sensor unit vectors and validity masks across modalities
    stacked = torch.stack(vectors, dim=1)  # (B, J, D)
    weight = torch.stack(held, dim=1).unsqueeze(-1).to(stacked.dtype)  # (B, J, 1)
    # Sum the instrument vectors the feature holds and normalise
    return functional.normalize((stacked * weight).sum(dim=1), dim=-1)  # (B, D)
