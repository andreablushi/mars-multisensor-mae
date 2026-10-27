"""Turning the patches one sensor was shown into tokens, one token a patch."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from architecture.components.acquisition_conditioning import AcquisitionConditioning
from architecture.components.cluttergram_conditioning import CluttergramConditioning
from architecture.components.transformer import Transformer
from architecture.tokens import Tokens

# B = batch, K = patches, D = token channels, P = patch dimensions.


class Encoder(nn.Module):
    """Embed each patch and attend over the set.

    Attributes:
        embed: From every sample of a patch to a token.
        conditioning: What tells a token its patch's condition, or None.
        acquisition: What tells a token how its patch was acquired, or None.
        cls: The token that leads every sequence and gathers it. (D)
        blocks: The transformer.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        condition: int,
        gain: bool,
        acquired: int,
        dim: int,
        heads: int,
        depth: int,
    ) -> None:
        """Build the encoder for one instrument.

        Args:
            shape: The shape of one patch of the instrument.
            condition: How many values each patch is conditioned on.
            gain: Whether the condition also scales each token, as FiLM does.
            acquired: How many acquisition angles each patch carries.
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
        self.conditioning = (
            CluttergramConditioning(condition, dim, gain) if condition else None
        )
        self.acquisition = AcquisitionConditioning(acquired, dim) if acquired else None

    def forward(self, patches: Tokens, visible: Tensor) -> Tensor:
        """Return the encoded tokens.

        Args:
            patches: The instrument's patches over the batch.
            visible: Which patches the encoder may read. (B, K)

        Returns:
            tokens: The CLS token, then one per slot, meaningful where visible.
                (B, 1 + K, D)
        """
        # One token per patch, from its measured samples and its condition
        read = (patches.values * patches.measured).flatten(2)  # (B, K, prod P)
        keys = self.embed(read)  # (B, K, D)
        if self.conditioning is not None:
            keys = self.conditioning(keys, patches.condition)
        if self.acquisition is not None:
            keys = self.acquisition(keys, patches.acquisition)
        keys = torch.cat([self.cls.expand(len(keys), 1, -1), keys], 1)  # (B, 1 + K, D)
        # Attend over the visible ones, placed on the ground by the rotary encoding
        readable, position = patches.readable_and_position_with_cls(visible)
        return checkpoint(
            self.blocks, keys, readable.unsqueeze(1), position, use_reentrant=False
        )  # (B, 1 + K, D)
