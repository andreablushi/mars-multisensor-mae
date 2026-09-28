"""Turning the patches one sensor was shown into tokens, one token a patch."""

from __future__ import annotations

import math
from collections.abc import Sequence

from building.common.layout import Axis
from torch import Tensor, nn
from torch.nn import functional

from architecture.components.channel_encoding import ChannelEncoding
from architecture.components.positional_encoding import PositionalEncoding
from architecture.components.transformer import Transformer

# B = batch, K = patches, C = channels, G = channel samples, D = token channels,
# P = patch dimensions.


class Encoder(nn.Module):
    """Embed each patch by channel, place it on the ground, and attend over the set.

    Attributes:
        at: The wavelength axis, if the patch has one.
        embed: From one channel's samples to a token.
        channels: The channel encoding.
        place: The positional encoding.
        blocks: The transformer.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        axes: tuple[str, ...],
        centres_nm: Sequence[float] | None,
        dim: int,
        heads: int,
        depth: int,
        stride: float,
    ) -> None:
        """Build the encoder for one instrument.

        Args:
            shape: The shape of one patch of the instrument.
            axes: What each of its axes holds, in that same order.
            centres_nm: The wavelength each channel is centred on, in nm, or None.
            dim: The token width.
            heads: How many attention heads each block runs.
            depth: How many blocks are stacked.
            stride: How far apart two neighbouring patch centres sit, in metres.
        """
        super().__init__()
        self.at = axes.index(Axis.WAVELENGTH) if Axis.WAVELENGTH in axes else None
        # Map one channel of a patch into the token width
        self.embed = nn.Linear(
            math.prod(size for at, size in enumerate(shape) if at != self.at), dim
        )
        self.channels = ChannelEncoding(dim, centres_nm)
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
            valid: Whether each sample is a measurement, broadcastable. (B, K, *P')
            position: Where each patch sits and how far it reaches, in metres. (B, K, 6)
            visible: Which patches the encoder may read. (B, K)

        Returns:
            tokens: One per slot, meaningful where visible. (B, K, D)
        """
        if self.at is None:
            values, valid = values.unsqueeze(-1), valid.unsqueeze(-1)
        else:
            values = values.movedim(2 + self.at, -1)
            valid = valid.movedim(2 + self.at, -1)
        bands = values.flatten(2, -2).transpose(2, 3)  # (B, K, C, G)
        measured = valid.flatten(2, -2).any(dim=2).unsqueeze(-1)  # (B, K, C, 1)
        tokens = functional.gelu(self.channels(self.embed(bands)))  # (B, K, C, D)
        counted = measured.sum(dim=2).clamp(min=1)  # (B, K, 1)
        tokens = (tokens * measured).sum(dim=2) / counted  # (B, K, D)
        # Place the tokens on the ground and attend over the visible ones
        return self.blocks(tokens + self.place(position), visible)  # (B, K, D)
