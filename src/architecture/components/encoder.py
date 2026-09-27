"""Turning the patches one sensor was shown into tokens, one token a patch."""

from __future__ import annotations

import math

from building.common.layout import Axis
from torch import Tensor, nn

from architecture.components.positional_encoding import PositionalEncoding
from architecture.components.transformer import Transformer

# B = batch, K = patches, D = token channels, P = patch dimensions.


class Encoder(nn.Module):
    """Embed each patch, place it on the ground, and attend over the set.

    Attributes:
        at: The wavelength axis, if the patch has one.
        embed: From patch samples to a token.
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
        self.at = axes.index(Axis.WAVELENGTH) if Axis.WAVELENGTH in axes else None
        # Map one patch, or the mean CRISM band, into the token width
        self.embed = nn.Linear(
            math.prod(size for at, size in enumerate(shape) if at != self.at), dim
        )
        # Module for continuous geospatial (metric coordinate) positional embeddings
        self.place = PositionalEncoding(dim, stride)
        # Transformer encoder stack for intra-sensor self-attention
        self.blocks = Transformer(dim, heads, depth)

    def forward(
        self,
        values: Tensor,
        position: Tensor,
        visible: Tensor,
    ) -> Tensor:
        """Return the encoded tokens.

        Args:
            values: The normalised patches. (B, K, *P)
            position: Where each patch sits and how far it reaches, in metres. (B, K, 6)
            visible: Which patches the encoder may read. (B, K)

        Returns:
            tokens: One per slot, meaningful where visible. (B, K, D)
        """
        if self.at is not None:
            values = values.mean(dim=2 + self.at)
        tokens = self.embed(values.flatten(2))
        # Place the tokens on the ground and attend over the visible ones
        return self.blocks(tokens + self.place(position), visible)  # (B, K, D)
