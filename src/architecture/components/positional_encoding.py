"""Where a token sits, as sines and cosines of its ground and height metres."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

SHORTEST_M = (100.0, 100.0, 10.0)
LONGEST_M = (1_000_000.0, 1_000_000.0, 30_000.0)


class PositionalEncoding(nn.Module):
    """A fixed Fourier encoding of east, north and height, a third of the width each.

    Attributes:
        wavelengths: The wavelengths each coordinate is read at, spaced evenly
            in the log from the shortest to the longest an instrument resolves.
            (3, D / 6)
    """

    def __init__(self, dim: int) -> None:
        """Lay out the wavelengths for one token width.

        Args:
            dim: The token width, a multiple of 6 so that each of the three
                coordinates gets a cosine and a sine per wavelength.
        """
        super().__init__()
        if dim % 6:
            raise ValueError(f"the token width must be a multiple of 6, not {dim}")
        wavelengths = torch.stack(
            [
                torch.logspace(math.log10(shortest), math.log10(longest), dim // 6)
                for shortest, longest in zip(SHORTEST_M, LONGEST_M, strict=True)
            ]
        )  # (3, D / 6)
        self.register_buffer("wavelengths", wavelengths)

    def forward(self, position: Tensor) -> Tensor:
        """Return the encoding of every position.

        Args:
            position: East, north and height of every patch centre, in metres.
                (B, K, 3)

        Returns:
            encoded: The cosines then the sines of each coordinate over its
                wavelengths, the three coordinates side by side. (B, K, D)
        """
        phase = (
            2 * math.pi * position.unsqueeze(-1) / self.wavelengths
        )  # (B, K, 3, D/6)
        encoded = torch.cat([phase.cos(), phase.sin()], dim=-1)  # (B, K, 3, D / 3)
        return encoded.flatten(-2)  # (B, K, D)
