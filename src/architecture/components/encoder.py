"""One instrument's encoder, a transformer over the patches it is shown."""

from __future__ import annotations

import math

from torch import Tensor, nn

from architecture.components.positional_encoding import PositionalEncoding
from architecture.components.transformer import Transformer


class Encoder(nn.Module):
    """One instrument's encoder: embed each patch, place it, attend over the visible.

    Attributes:
        embed: From every value of a patch to one token.
        place: The positional encoding.
        blocks: The transformer.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        dim: int,
        heads: int,
        depth: int,
        resolution: float,
    ) -> None:
        """Build the encoder for one instrument.

        Args:
            shape: The shape of one patch of the instrument.
            dim: The token width.
            heads: How many attention heads each block runs.
            depth: How many blocks are stacked.
            resolution: How much ground one sample of the instrument spans, in
                metres.
        """
        super().__init__()
        self.embed = nn.Linear(math.prod(shape), dim)
        self.place = PositionalEncoding(dim, resolution)
        self.blocks = Transformer(dim, heads, depth)

    def forward(self, values: Tensor, position: Tensor, visible: Tensor) -> Tensor:
        """Return the encoded tokens.

        Args:
            values: The normalised patches. (B, K, *P)
            position: Where each patch sits and how far it reaches, in metres.
                (B, K, 6)
            visible: Which patches the encoder may read. (B, K)

        Returns:
            tokens: One per slot, meaningful where visible. (B, K, D)
        """
        tokens = self.embed(values.flatten(2)) + self.place(position)  # (B, K, D)
        return self.blocks(tokens, visible)  # (B, K, D)
