"""The latent space: every token a direction on the unit sphere, drawn about a mean."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional


class Sphere(nn.Module):
    """Placing tokens on the unit sphere, and drawing about them while training.

    Attributes:
        project: From the encoder width down to the sphere's dimension.
        kappa: How tightly a draw stays about its mean direction.
    """

    def __init__(self, dim: int, latent: int, kappa: float) -> None:
        """Size the sphere for one token width.

        Args:
            dim: The token width coming in.
            latent: The sphere's dimension.
            kappa: How tightly a draw stays about its mean direction.
        """
        super().__init__()
        self.project = nn.Linear(dim, latent)
        self.kappa = kappa

    def forward(self, tokens: Tensor) -> Tensor:
        """Return every token as a unit vector.

        Args:
            tokens: The tokens as the cross-sensor encoder hands them. (B, K, D)

        Returns:
            directions: Unit vectors, the mean direction of each token when
                evaluating and a draw about it when training. (B, K, L)
        """
        mean = functional.normalize(self.project(tokens), dim=-1)  # (B, K, L)
        if not self.training:
            return mean
        drawn = mean + torch.randn_like(mean) / math.sqrt(self.kappa)  # (B, K, L)
        return functional.normalize(drawn, dim=-1)  # (B, K, L)


def mean_direction(directions: Tensor, present: Tensor) -> Tensor:
    """Return where a set of unit vectors points on average.

    Args:
        directions: Unit vectors. (B, K, L)
        present: Which of them count. (B, K)

    Returns:
        mean: The unit vector along their sum, zero where none counts. (B, L)
    """
    weight = present.unsqueeze(-1).to(directions.dtype)  # (B, K, 1)
    summed = (directions * weight).sum(dim=1)  # (B, L)
    return functional.normalize(summed, dim=-1)  # (B, L)
