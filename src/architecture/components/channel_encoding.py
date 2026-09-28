"""Saying which wavelength a channel of a patch measures, read as sinusoids."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

# B = batch, K = patches, C = channels, D = token channels.

SHORTEST_DECADES = 1e-3

LONGEST_DECADES = 10.0


class ChannelEncoding(nn.Module):
    """A fixed Fourier encoding of each channel's wavelength, on a log scale.

    An instrument without a wavelength axis reads its patch as one channel,
    encoded as zero.

    Attributes:
        encoded: The cosines then sines of each channel's wavelength. (C, D)
    """

    def __init__(self, dim: int, centres_nm: Sequence[float] | None) -> None:
        """Lay out the encoding of every channel of one instrument.

        Args:
            dim: The token width, a cosine and a sine per period.
            centres_nm: The wavelength each channel is centred on, in nm, or None.
        """
        super().__init__()
        periods = torch.logspace(
            math.log10(SHORTEST_DECADES), math.log10(LONGEST_DECADES), dim // 2
        )  # (D / 2)
        if centres_nm is None:
            encoded = torch.zeros(1, dim)  # (1, D)
        else:
            decades = torch.tensor(centres_nm, dtype=torch.float64).log10()  # (C)
            phase = 2 * math.pi * decades.unsqueeze(-1) / periods  # (C, D / 2)
            encoded = torch.cat([phase.cos(), phase.sin()], dim=-1).float()  # (C, D)
        self.register_buffer("encoded", encoded)

    def forward(self, tokens: Tensor) -> Tensor:
        """Return each channel's tokens with its wavelength added.

        Args:
            tokens: One token per channel of each patch. (B, K, C, D)

        Returns:
            tokens: The same tokens, each told its channel. (B, K, C, D)
        """
        return tokens + self.encoded  # (B, K, C, D)
