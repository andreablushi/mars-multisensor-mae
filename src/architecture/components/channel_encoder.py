"""Saying what each channel of a sensor measures, as one vector."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

# B = batch, K = patches, C = channels, D = token channels.

SHORTEST_NM = 6.0

LONGEST_NM = 2700.0


class ChannelEncoder(nn.Module):
    """A Fourier encoding of each channel's wavelength.

    An instrument without a wavelength axis reads its patch as one channel,
    encoded as zeros.

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
        # Wavelength periods spaced evenly in log, from the shortest to the longest
        periods = torch.logspace(
            math.log10(SHORTEST_NM), math.log10(LONGEST_NM), dim // 2
        )  # (D / 2)
        # An instrument with no wavelengths has one channel, encoded as zeros
        if centres_nm is None:
            encoded = torch.zeros(1, dim)  # (1, D)
        else:
            # Each channel's centre wavelength, in double precision for the phases
            centres = torch.tensor(centres_nm, dtype=torch.float64)  # (C)
            # The phase of each wavelength against each period
            phase = 2 * math.pi * centres.unsqueeze(-1) / periods  # (C, D / 2)
            # Cosines then sines of those phases, one vector per channel
            encoded = torch.cat([phase.cos(), phase.sin()], dim=-1).float()  # (C, D)
        # Kept with the model and moved with it, but never trained
        self.register_buffer("encoded", encoded)

    def forward(self, tokens: Tensor) -> Tensor:
        """Return each channel's tokens with its wavelength added.

        Args:
            tokens: One token per channel of each patch. (B, K, C, D)

        Returns:
            tokens: The same tokens, each told its channel. (B, K, C, D)
        """
        # Add each channel's wavelength encoding
        return tokens + self.encoded  # (B, K, C, D)
