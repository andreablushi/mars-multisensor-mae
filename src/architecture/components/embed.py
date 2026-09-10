"""Turning one instrument's patches into tokens."""

from __future__ import annotations

import math

from torch import Tensor, nn


class PatchEmbedding(nn.Module):
    """A linear map from every value of a patch to one token.

    Attributes:
        project: The map itself.
    """

    def __init__(self, shape: tuple[int, ...], dim: int) -> None:
        """Size the map for one instrument's patches.

        Args:
            shape: The shape of one patch of the instrument.
            dim: The token width.
        """
        super().__init__()
        self.project = nn.Linear(math.prod(shape), dim)

    def forward(self, values: Tensor) -> Tensor:
        """Return one token per patch.

        Args:
            values: The normalised patches. (B, K, *P)

        Returns:
            tokens: One per patch. (B, K, D)
        """
        flat = values.flatten(2)  # (B, K, prod(P))
        return self.project(flat)  # (B, K, D)
