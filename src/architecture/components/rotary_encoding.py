"""Saying where a token sits by rotating what it asks and answers with."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

# B = batch, H = heads, N = tokens, E = head channels, F = frequencies.

SHORTEST_M = 1280.0

LONGEST_M = 64_000.0

COORDINATES = 2


class RotaryEncoding(nn.Module):
    """A rotary encoding of east and north, so attention reads ground offsets alone.

    Attributes:
        frequencies: The angular frequency of each rotated pair, in radians a metre. (F)
    """

    def __init__(self, head_dim: int) -> None:
        """Lay out the frequencies for one head width.

        Args:
            head_dim: How wide one attention head is, its remainder past 4F left as is.
        """
        super().__init__()
        wavelengths = torch.logspace(
            math.log10(SHORTEST_M), math.log10(LONGEST_M), head_dim // (2 * COORDINATES)
        )  # (F)
        self.register_buffer("frequencies", 2 * math.pi / wavelengths)

    def forward(self, heads: Tensor, position: Tensor) -> Tensor:
        """Return the heads rotated by where each token sits.

        Args:
            heads: The queries or keys of every head. (B, H, N, E)
            position: Each token's patch centre, east and north in metres first, a
                sounder's delay after them left out. (B, N, 3)

        Returns:
            rotated: The same heads, each pair turned by its coordinate. (B, H, N, E)
        """
        where = position[..., :COORDINATES]  # (B, N, 2)
        angle = (where.unsqueeze(-1) * self.frequencies).flatten(-2)  # (B, N, 2F)
        cos = angle.cos().unsqueeze(1).to(heads.dtype)  # (B, 1, N, 2F)
        sin = angle.sin().unsqueeze(1).to(heads.dtype)  # (B, 1, N, 2F)
        turned = angle.shape[-1]
        first = heads[..., :turned]  # (B, H, N, 2F)
        second = heads[..., turned : 2 * turned]  # (B, H, N, 2F)
        return torch.cat(
            [
                first * cos - second * sin,
                first * sin + second * cos,
                heads[..., 2 * turned :],
            ],
            dim=-1,
        )  # (B, H, N, E)
