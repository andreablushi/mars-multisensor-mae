"""Saying where a token sits by rotating what it asks and answers with."""

from __future__ import annotations

import math

import torch
from mars_multisensor_dataset.building.configs import sharad
from mars_multisensor_dataset.common.maths import physics
from torch import Tensor, nn

SHORTEST_M = 1280.0

LONGEST_M = 64_000.0

METRES_PER_ROW = physics.SPEED_OF_LIGHT_M_S * sharad.DELAY_INTERVAL_S / 2

DELAY_LONGEST_M = sharad.DELAY_ROWS * METRES_PER_ROW

DELAY_PAIRS = 4


class RotaryEncoding(nn.Module):
    """A rotary encoding of east, north and height, so attention reads offsets alone.

    Attributes:
        frequencies: The angular frequency of each rotated pair, in radians a metre. (P)
        axis: Which coordinate each pair is turned by, east, north or height. (P)
    """

    def __init__(self, head_dim: int) -> None:
        """Lay out the frequencies for one head width.

        Args:
            head_dim: How wide one attention head is, DELAY_PAIRS of its pairs turned
                by height over the delay window and the rest shared by east and north.
        """
        super().__init__()
        ground = (head_dim // 2 - DELAY_PAIRS) // 2
        spans = (
            (ground, LONGEST_M),
            (ground, LONGEST_M),
            (DELAY_PAIRS, DELAY_LONGEST_M),
        )
        wavelengths = torch.cat(
            [
                torch.logspace(math.log10(SHORTEST_M), math.log10(longest), pairs)
                for pairs, longest in spans
            ]
        )  # (P)
        self.register_buffer("frequencies", 2 * math.pi / wavelengths)
        self.register_buffer(
            "axis",
            torch.arange(len(spans)).repeat_interleave(
                torch.tensor([pairs for pairs, _ in spans])
            ),
        )

    def forward(self, heads: Tensor, position: Tensor) -> Tensor:
        """Return the heads rotated by where each token sits.

        Args:
            heads: The queries or keys of every head. (B, H, N, E)
            position: Each token's patch centre, in metres or rows. (B, N, 3)

        Returns:
            rotated: The same heads, each pair turned by its coordinate. (B, H, N, E)
        """
        where = torch.stack(
            [position[..., 0], position[..., 1], position[..., 2] * METRES_PER_ROW],
            dim=-1,
        )  # (B, N, 3)
        angle = where[..., self.axis] * self.frequencies  # (B, N, P)
        cos = angle.cos().unsqueeze(1).to(heads.dtype)  # (B, 1, N, P)
        sin = angle.sin().unsqueeze(1).to(heads.dtype)  # (B, 1, N, P)
        turned = angle.shape[-1]
        first = heads[..., :turned]  # (B, H, N, P)
        second = heads[..., turned : 2 * turned]  # (B, H, N, P)
        return torch.cat(
            [
                first * cos - second * sin,
                first * sin + second * cos,
                heads[..., 2 * turned :],
            ],
            dim=-1,
        )  # (B, H, N, E)
