"""Telling a sounder's token how deep its patch sits, apart from where on the ground."""

from __future__ import annotations

import math

import torch
from building.configs import sharad
from common.maths import physics
from torch import Tensor, nn

from architecture.components.rotary_encoding import LONGEST_M, SHORTEST_M

# B = batch, K = patches, D = token channels, F = frequencies.

METRES_PER_ROW = physics.SPEED_OF_LIGHT_M_S * sharad.DELAY_INTERVAL_S / 2


class DelayEncoding(nn.Module):
    """Add a sine and cosine code of each patch's delay to its token.

    Attributes:
        frequencies: The angular frequency of each pair, in radians a metre. (F)
    """

    def __init__(self, dim: int) -> None:
        """Lay out the frequencies for one token width.

        Args:
            dim: The token width, half of it sines and half cosines.
        """
        super().__init__()
        wavelengths = torch.logspace(
            math.log10(SHORTEST_M), math.log10(LONGEST_M), dim // 2
        )  # (F)
        self.register_buffer("frequencies", 2 * math.pi / wavelengths)

    def forward(self, tokens: Tensor, delay: Tensor) -> Tensor:
        """Return the tokens told how deep their patches sit.

        Args:
            tokens: One token per patch. (B, K, D)
            delay: Each patch's centre, in rows of delay. (B, K)

        Returns:
            tokens: The same tokens, coded. (B, K, D)
        """
        angle = (delay * METRES_PER_ROW).unsqueeze(-1) * self.frequencies  # (B, K, F)
        return tokens + torch.cat([angle.sin(), angle.cos()], -1).to(tokens.dtype)
