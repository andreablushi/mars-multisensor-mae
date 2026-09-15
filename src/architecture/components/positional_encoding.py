"""Saying where a patch sits and how far it reaches, in metres read as sinusoids."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

# How high above the areoid the ground runs, which no instrument settles.
HEIGHT_M = (10.0, 30_000.0)

# The widest a feature runs, which is what the longest ground period reads.
GROUND_LONGEST_M = 1_000_000.0

COORDINATES = 6


class PositionalEncoding(nn.Module):
    """A fixed Fourier encoding of east, north and height, of the centre and the span.

    A patch says where it sits and how far it reaches, so a surface tile, which
    spans ground and no height, and a sounding column, which spans height and
    one track, are told apart without either being named to the model.

    Attributes:
        periods: What each sinusoid repeats over, from the stride up. (6, D / 12)
    """

    def __init__(self, dim: int, stride: float) -> None:
        """Lay out the periods for one instrument at one token width.

        Args:
            dim: The token width, a multiple of 12, a cosine and a sine per coordinate.
            stride: How far apart two neighbouring patch centres sit, in metres.
        """
        super().__init__()
        # The width must hold a cosine and a sine for each coordinate
        if dim % (2 * COORDINATES):
            raise ValueError(f"the token width must be a multiple of 12, not {dim}")
        # Set minimum frequency bounds (stride for ground, 10m for height)
        shortest = (stride, stride, HEIGHT_M[0])
        # Set maximum frequency bounds (1000km for ground, 30km for height)
        longest = (GROUND_LONGEST_M, GROUND_LONGEST_M, HEIGHT_M[1])
        # Generate logarithmically spaced sinusoid periods for 3D centers and 3D extents
        periods = torch.stack(
            [
                torch.logspace(
                    math.log10(short), math.log10(long), dim // (2 * COORDINATES)
                )
                for short, long in zip(shortest * 2, longest * 2, strict=True)
            ]
        )  # (6, D / 12)
        self.register_buffer("periods", periods)

    def forward(self, position: Tensor) -> Tensor:
        """Return the encoding of every position.

        Args:
            position: The centre east, north and height, then each span. (B, K, 6)

        Returns:
            encoded: The cosines then sines of each coordinate, side by side. (B, K, D)
        """
        # The phase of each coordinate against each of its periods
        phase = 2 * math.pi * position.unsqueeze(-1) / self.periods  # (B, K, 6, D/12)
        # Compute cosine and sine harmonic pairs for each coordinate axis
        encoded = torch.cat([phase.cos(), phase.sin()], dim=-1)  # (B, K, 6, D / 6)
        # Flatten coordinate components into a unified D-dimensional positional vector
        return encoded.flatten(-2)  # (B, K, D)
