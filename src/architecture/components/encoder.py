"""Turning the patches one sensor was shown into tokens, one token a patch."""

from __future__ import annotations

import math

from torch import Tensor, nn

from architecture.components.span_encoding import SpanEncoding
from architecture.components.transformer import Transformer
from architecture.tokens import Tokens

# B = batch, K = patches, D = token channels, P = patch dimensions.


class Encoder(nn.Module):
    """Embed each patch, place it on the ground, and attend over the set.

    Attributes:
        embed: From every sample of a patch to a token.
        span: The encoding of how far each patch reaches.
        blocks: The transformer.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        dim: int,
        heads: int,
        depth: int,
        stride: float,
        radius: float,
    ) -> None:
        """Build the encoder for one instrument.

        Args:
            shape: The shape of one patch of the instrument.
            dim: The token width.
            heads: How many attention heads each block runs.
            depth: How many blocks are stacked.
            stride: How far apart two neighbouring patch centres sit, in metres.
            radius: How far apart on the ground two tokens may attend, in metres.
        """
        super().__init__()
        # Map every sample of a patch into the token width
        self.embed = nn.Linear(math.prod(shape), dim)
        # Encode how far a patch reaches, at the patch spacing
        self.span = SpanEncoding(dim, stride)
        # The transformer the instrument's patch tokens attend over each other in
        self.blocks = Transformer(dim, heads, depth, radius)

    def forward(self, patches: Tokens, visible: Tensor) -> Tensor:
        """Return the encoded tokens.

        Args:
            patches: The instrument's patches over the batch.
            visible: Which patches the encoder may read. (B, K)

        Returns:
            tokens: One per slot, meaningful where visible. (B, K, D)
        """
        # One token per patch, from its measured samples alone
        tokens = self.embed((patches.values * patches.measured).flatten(2))  # (B, K, D)
        # Place the tokens on the ground and attend over the visible ones
        return self.blocks(
            tokens + self.span(patches.position), visible, patches.position
        )  # (B, K, D)
