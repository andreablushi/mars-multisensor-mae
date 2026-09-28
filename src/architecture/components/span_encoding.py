"""Saying how far a patch reaches, read as sinusoids."""

from __future__ import annotations

import math

import torch
from building.configs import sharad
from torch import Tensor, nn

# B = batch, K = positions, D = token channels.

# The delay a patch spans, from one row of the radargram window to the whole of it.
DELAY_ROWS = (1.0, float(sharad.DELAY_ROWS))

GROUND_LONGEST_M = 64_000.0

COORDINATES = 3


class SpanEncoding(nn.Module):
    """A fixed Fourier encoding of how far a patch reaches east, north and in delay.

    A surface tile, which spans ground and no delay, and a sounding column, which
    spans delay and one track, are told apart without either being named to the
    model. Where a patch sits is left to the grid its cells are read into.

    Attributes:
        periods: What each sinusoid repeats over, from the stride up. (3, D / 6)
    """

    def __init__(self, dim: int, stride: float) -> None:
        """Lay out the periods for one instrument at one token width.

        Args:
            dim: The token width, a multiple of 6, a cosine and a sine per coordinate.
            stride: How far apart two neighbouring patch centres sit, in metres.
        """
        super().__init__()
        shortest = (stride, stride, DELAY_ROWS[0])
        longest = (GROUND_LONGEST_M, GROUND_LONGEST_M, DELAY_ROWS[1])
        periods = torch.stack(
            [
                torch.logspace(
                    math.log10(short), math.log10(long), dim // (2 * COORDINATES)
                )
                for short, long in zip(shortest, longest, strict=True)
            ]
        )  # (3, D / 6)
        self.register_buffer("periods", periods)

    def forward(self, position: Tensor) -> Tensor:
        """Return the encoding of every span.

        Args:
            position: The centre east, north and delay, then each span. (B, K, 6)

        Returns:
            encoded: The cosines then sines of each span, side by side. (B, K, D)
        """
        # The phase of each span against each of its periods
        phase = (
            2 * math.pi * position[..., 3:].unsqueeze(-1) / self.periods
        )  # (B, K, 3, D/6)
        encoded = torch.cat([phase.cos(), phase.sin()], dim=-1)  # (B, K, 3, D / 3)
        return encoded.flatten(-2)  # (B, K, D)
