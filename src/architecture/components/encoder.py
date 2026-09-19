"""Turning the patches one sensor was shown into tokens, one token a patch."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from architecture.components.positional_encoding import PositionalEncoding
from architecture.components.transformer import Transformer
from dataset.patches import channel_axis


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


class Encoder(nn.Module):
    """Embed each channel of a patch, place it on the ground, attend over the set.

    Attributes:
        at: Which axis a patch's channels run along, or None where it holds one.
        embed: From one channel's ground samples to one token, shared by all.
        channel: What each channel of this instrument is, one vector each. (C, D)
        place: The positional encoding.
        blocks: The transformer.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        axes: tuple[str, ...],
        dim: int,
        heads: int,
        depth: int,
        stride: float,
    ) -> None:
        """Build the encoder for one instrument.

        Args:
            shape: The shape of one patch of the instrument.
            axes: What each of its axes holds, in that same order.
            dim: The token width.
            heads: How many attention heads each block runs.
            depth: How many blocks are stacked.
            stride: How far apart two neighbouring patch centres sit, in metres.
        """
        super().__init__()
        # Identify which axis index corresponds to sensor channels
        self.at = channel_axis(axes)
        # Map one channel's ground samples into the token width
        self.embed = nn.Linear(
            math.prod(size for at, size in enumerate(shape) if at != self.at), dim
        )
        # One vector per channel, since every crop is laid out on the one fixed grid
        self.channel = nn.Parameter(
            torch.zeros(shape[self.at] if self.at is not None else 1, dim)
        )  # (C, D)
        nn.init.normal_(self.channel, std=0.02)
        # Module for continuous geospatial (metric coordinate) positional embeddings
        self.place = PositionalEncoding(dim, stride)
        # Transformer encoder stack for intra-sensor self-attention
        self.blocks = Transformer(dim, heads, depth)

    def forward(
        self,
        values: Tensor,
        valid: Tensor,
        position: Tensor,
        visible: Tensor,
    ) -> Tensor:
        """Return the encoded tokens.

        Args:
            values: The normalised patches. (B, K, *P)
            valid: Whether each sample of a patch is a measurement. (B, K, *P')
            position: Where each patch sits and how far it reaches, in metres. (B, K, 6)
            visible: Which patches the encoder may read. (B, K)

        Returns:
            tokens: One per slot, meaningful where visible. (B, K, D)
        """
        # Embed each channel's samples and add which channel of the grid it is
        held = self.embed(by_channel(values, self.at)) + self.channel  # (B, K, C, D)
        # Mark which channels hold a measurement
        weight = by_channel(valid, self.at).any(dim=-1).unsqueeze(-1)  # (B, K, C', 1)
        weight = weight.to(held.dtype)  # (B, K, C', 1)
        # Pool the channels that hold one into a single patch token
        tokens = (held * weight).sum(dim=2) / weight.sum(dim=2).clamp(
            min=1
        )  # (B, K, D)
        # Place the tokens on the ground and attend over the visible ones
        return self.blocks(tokens + self.place(position), visible)  # (B, K, D)
