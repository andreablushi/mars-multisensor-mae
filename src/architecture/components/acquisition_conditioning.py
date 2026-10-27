"""Telling a token how the observation its patch belongs to was acquired."""

from __future__ import annotations

import torch
from torch import Tensor, nn

# B = batch, K = patches, A = acquisition fields, D = token channels.


class AcquisitionConditioning(nn.Module):
    """Add each patch's acquisition angles, mapped to the token width, to its token.

    Attributes:
        project: From the sine and cosine of every angle to the token width, started
            at zero so a conditioned model starts as an unconditioned one.
    """

    def __init__(self, fields: int, dim: int) -> None:
        """Build the conditioning for one token width.

        Args:
            fields: How many angles each patch carries.
            dim: The token width.
        """
        super().__init__()
        self.project = nn.Linear(2 * fields, dim, bias=False)
        nn.init.zeros_(self.project.weight)

    def forward(self, tokens: Tensor, degrees: Tensor) -> Tensor:
        """Return the tokens told how their patches were acquired.

        Args:
            tokens: One token per patch. (B, K, D)
            degrees: Each patch's acquisition angles, NaN where missing, which reads
                as a zero sine and cosine no angle has. (B, K, A)

        Returns:
            tokens: The same tokens, conditioned. (B, K, D)
        """
        angles = torch.deg2rad(degrees)
        encoded = torch.cat([angles.sin(), angles.cos()], -1).nan_to_num()
        return tokens + self.project(encoded)
