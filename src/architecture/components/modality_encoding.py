"""Saying what a channel measures, which the numbers it holds do not."""

from __future__ import annotations

import math

import torch
from building.common.layout import ELEVATION, WAVELENGTH
from torch import Tensor, nn

from dataset.patches import channel_axis

CHANNEL_RUN = {WAVELENGTH: (6.0, 2700.0), ELEVATION: (11.0, 30_000.0)}


def by_channel(held: Tensor, at: int | None) -> Tensor:
    """Return one batch of patches as the ground samples of each of their channels.

    Args:
        held: One batch of patches, or of their validity. (B, K, *P)
        at: Which axis the channels run along, or None where a patch holds one.

    Returns:
        samples: The ground samples of each channel. (B, K, C, G)
    """
    if at is None:
        # Reshape single-channel patch to (B, K, 1, G) with single channel dimension
        return held.flatten(2).unsqueeze(2)  # (B, K, 1, G)
    # Move the channel axis forward and flatten the ground samples
    return held.movedim(2 + at, 2).flatten(3)  # (B, K, C, G)


class ModalityEncoding(nn.Module):
    """Say what a sensor is and what each channel of it measures, as one vector."""

    def __init__(self, axes: tuple[str, ...], dim: int) -> None:
        """Build the encoding for one instrument at one token width.

        Args:
            axes: What each axis of the instrument's patches holds.
            dim: The token width.
        """
        super().__init__()
        # Learnable embedding vector identifying the specific instrument/sensor type
        self.mark = nn.Parameter(torch.zeros(dim))  # (D)
        nn.init.normal_(self.mark, std=0.02)
        # Identify physical measurement axis (e.g., wavelength or elevation)
        at = channel_axis(axes)
        run = CHANNEL_RUN.get(axes[at]) if at is not None else None
        # Handle panchromatic/single-channel inputs without continuous physical scales
        if run is None:
            self.periods = None
            return
        shortest, longest = run
        # Log-spaced periods, from the shortest step to the whole range
        self.register_buffer(
            "periods",
            torch.logspace(math.log10(shortest), math.log10(longest), dim // 2),
        )  # (D / 2)

    def forward(self, channels: Tensor) -> Tensor:
        """Return what each channel of each patch is.

        Args:
            channels: What each channel of each patch measures. (B, K, C)

        Returns:
            encoded: The instrument mark, plus each channel's sinusoids. (B, K, C, D)
        """
        # The instrument mark alone where its channels run on no scale
        if self.periods is None:
            return self.mark.expand(*channels.shape, -1)  # (B, K, C, D)
        # Compute phase angles: phase = 2 * pi * channel_value / period
        phase = 2 * math.pi * channels.unsqueeze(-1) / self.periods  # (B, K, C, D / 2)
        # The cosines then the sines, with the instrument mark added
        return self.mark + torch.cat([phase.cos(), phase.sin()], dim=-1)  # (B, K, C, D)
