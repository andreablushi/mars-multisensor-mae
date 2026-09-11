"""Where a token sits and how far it reaches, as sines and cosines of its metres."""

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
        periods: The ground distance each sinusoid repeats over, spaced
            evenly in the log from the shortest the instrument resolves to the
            widest a feature runs, a span read at the same ones as the
            coordinate it spans. (6, D / 12)
    """

    def __init__(self, dim: int, resolution: float) -> None:
        """Lay out the periods for one instrument at one token width.

        Args:
            dim: The token width, a multiple of 12 so that each of the six
                coordinates gets a cosine and a sine per period.
            resolution: How much ground one sample of the instrument spans, in
                metres, which is the shortest it can tell anything apart at.
        """
        super().__init__()
        # TODO: remove this check since I know what I'm passing to it
        if dim % (2 * COORDINATES):
            raise ValueError(f"the token width must be a multiple of 12, not {dim}")
        shortest = (resolution, resolution, HEIGHT_M[0])
        longest = (GROUND_LONGEST_M, GROUND_LONGEST_M, HEIGHT_M[1])
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
            position: East, north and height of every patch centre, then how
                far it spans along each of the three, in metres. (B, K, 6)

        Returns:
            encoded: The cosines then the sines of each coordinate over its
                periods, the six coordinates side by side. (B, K, D)
        """
        phase = 2 * math.pi * position.unsqueeze(-1) / self.periods  # (B, K, 6, D/12)
        encoded = torch.cat([phase.cos(), phase.sin()], dim=-1)  # (B, K, 6, D / 6)
        return encoded.flatten(-2)  # (B, K, D)
