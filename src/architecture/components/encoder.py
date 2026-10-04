"""Turning the patches one sensor was shown into tokens, one token a patch."""

from __future__ import annotations

import math

from torch import Tensor, nn

from architecture.components.positional_encoding import PositionalEncoding
from architecture.components.transformer import Transformer

# B = batch, K = patches, D = token channels, P = patch dimensions.


class Encoder(nn.Module):
    """Embed each patch, place it on the ground, and attend over the set.

    Attributes:
        embed: From every sample of a patch to a token.
        place: The positional encoding.
        blocks: The transformer.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        dim: int,
        heads: int,
        depth: int,
        stride: float,
    ) -> None:
        """Build the encoder for one instrument.

        Args:
            shape: The shape of one patch of the instrument.
            dim: The token width.
            heads: How many attention heads each block runs.
            depth: How many blocks are stacked.
            stride: How far apart two neighbouring patch centres sit, in metres.
        """
        super().__init__()
        # Map every sample of a patch into the token width
        self.embed = nn.Linear(math.prod(shape), dim)
        # Encode where a patch sits and how far it reaches, at the patch spacing
        self.place = PositionalEncoding(dim, stride)
        # The transformer the instrument's patch tokens attend over each other in
        self.blocks = Transformer(dim, heads, depth)

    def forward(
        self, values: Tensor, measured: Tensor, position: Tensor, visible: Tensor
    ) -> Tensor:
        """Return the encoded tokens.

        Args:
            values: The normalised patches. (B, K, *P)
            measured: Whether each sample is a measurement, broadcastable. (B, K, *P')
            position: Where each patch sits and how far it reaches, in metres. (B, K, 6)
            visible: Which patches the encoder may read. (B, K)

        Returns:
            tokens: One per slot, meaningful where visible. (B, K, D)
        """
        # One token per patch, from its measured samples alone
        tokens = self.embed((values * measured).flatten(2))  # (B, K, D)
        # Place the tokens on the ground and attend over the visible ones
        return self.blocks(tokens + self.place(position), visible)  # (B, K, D)
