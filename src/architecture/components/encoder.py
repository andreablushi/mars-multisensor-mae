"""Turning the patches one sensor was shown into tokens, one token a patch."""

from __future__ import annotations

import math

from torch import Tensor, nn

from architecture.components.modalityencoder import ModalityEncoder, by_channel
from architecture.components.positional_encoding import PositionalEncoding
from architecture.components.transformer import Transformer
from dataset.patches import channel_axis


class Encoder(nn.Module):
    """Embed each channel of a patch, place it on the ground, attend over the set.

    Attributes:
        at: Which axis a patch's channels run along, or None where it holds one.
        embed: From one channel's ground samples to one token, shared by all.
        modality: What the instrument is and what each channel of it measures.
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
        self.at = channel_axis(axes)
        self.embed = nn.Linear(
            math.prod(size for at, size in enumerate(shape) if at != self.at), dim
        )
        self.modality = ModalityEncoder(axes, dim)
        self.place = PositionalEncoding(dim, stride)
        self.blocks = Transformer(dim, heads, depth)

    def forward(
        self,
        values: Tensor,
        channels: Tensor,
        valid: Tensor,
        position: Tensor,
        visible: Tensor,
    ) -> Tensor:
        """Return the encoded tokens.

        Args:
            values: The normalised patches. (B, K, *P)
            channels: What each channel of each patch measures. (B, K, C)
            valid: Whether each sample of a patch is a measurement. (B, K, *P')
            position: Where each patch sits and how far it reaches, in metres. (B, K, 6)
            visible: Which patches the encoder may read. (B, K)

        Returns:
            tokens: One per slot, meaningful where visible. (B, K, D)
        """
        held = self.embed(by_channel(values, self.at)) + self.modality(
            channels
        )  # (B, K, C, D)
        # A channel nothing measured was filled rather than read, so it is left out.
        weight = by_channel(valid, self.at).any(dim=-1).unsqueeze(-1)  # (B, K, C', 1)
        weight = weight.to(held.dtype)  # (B, K, C', 1)
        tokens = (held * weight).sum(dim=2) / weight.sum(dim=2).clamp(
            min=1
        )  # (B, K, D)
        return self.blocks(tokens + self.place(position), visible)  # (B, K, D)
