"""Turning the patches one sensor was shown into tokens, one token a patch."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from architecture.components.transformer import Transformer
from architecture.tokens import Tokens

# B = batch, K = patches, D = token channels, P = patch dimensions.


class Encoder(nn.Module):
    """Embed each patch and attend over the set.

    Attributes:
        embed: From every sample of a patch to a token.
        cls: The token that leads every sequence and gathers it. (D)
        blocks: The transformer.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        dim: int,
        heads: int,
        depth: int,
    ) -> None:
        """Build the encoder for one instrument.

        Args:
            shape: The shape of one patch of the instrument.
            dim: The token width.
            heads: How many attention heads each block runs.
            depth: How many blocks are stacked.
        """
        super().__init__()
        # Map every sample of a patch into the token width
        self.embed = nn.Linear(math.prod(shape), dim)
        # One learned CLS token, started small as transformer embeddings are
        self.cls = nn.Parameter(torch.zeros(dim))  # (D)
        nn.init.normal_(self.cls, std=0.02)
        # The transformer the instrument's patch tokens attend over each other in
        self.blocks = Transformer(dim, heads, depth)

    def forward(self, patches: Tokens, visible: Tensor) -> Tensor:
        """Return the encoded tokens.

        Args:
            patches: The instrument's patches over the batch.
            visible: Which patches the encoder may read. (B, K)

        Returns:
            tokens: The CLS token, then one per slot, meaningful where visible.
                (B, 1 + K, D)
        """
        # One token per patch, from its measured samples alone, each a key and a query
        keys = self.embed((patches.values * patches.measured).flatten(2))  # (B, K, D)
        keys = torch.cat([self.cls.expand(len(keys), 1, -1), keys], 1)  # (B, 1 + K, D)
        # Attend over the visible ones, placed on the ground by the rotary encoding
        return checkpoint(
            self.blocks, keys, *patches.led_by_cls(visible), use_reentrant=False
        )  # (B, 1 + K, D)
