"""Turning the patches one sensor was shown into tokens, one token a patch."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from mars_multisensor_mae.architecture.components.acquisition_conditioning import (
    AcquisitionConditioning,
)
from mars_multisensor_mae.architecture.components.cluttergram_conditioning import (
    CluttergramConditioning,
)
from mars_multisensor_mae.architecture.components.transformer import Transformer
from mars_multisensor_mae.architecture.models.tokens import Tokens


class Encoder(nn.Module):
    """Embed each patch and attend over the set.

    Attributes:
        embed: From every sample of a patch to a token.
        conditioning: What tells a token its patch's condition, or None.
        acquisition: What tells a token how its patch was acquired, or None.
        cls: The token that leads every sequence and gathers it. (D)
        blocks: The transformer.
        checkpointed: Whether the blocks' activations are recomputed in the backward
            pass instead of kept.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        condition: int,
        acquired: int,
        dim: int,
        heads: int,
        depth: int,
    ) -> None:
        """Build the encoder for one instrument.

        Args:
            shape: The shape of one patch of the instrument.
            condition: How many values each patch is conditioned on.
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
            CluttergramConditioning(condition, dim) if condition else None
        )
        self.acquisition = AcquisitionConditioning(acquired, dim) if acquired else None
        self.checkpointed = True

    def forward(self, patches: Tokens, readable: Tensor, position: Tensor) -> Tensor:
        """Return the encoded tokens.

        Args:
            patches: The instrument's patches over the batch.
            readable: Which slots the encoder may read, a CLS slot leading. (B, 1 + K)
            position: Each slot's patch centre, a CLS slot leading. (B, 1 + K, 3)

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
        # The CLS reads every patch and none reads it back, so patches see offsets alone
        heard = torch.ones(
            keys.shape[1], keys.shape[1], dtype=torch.bool, device=keys.device
        )
        heard[1:, 0] = False
        reads = readable.unsqueeze(1) & heard  # (B, 1 + K, 1 + K)
        if not self.checkpointed:
            return self.blocks(keys, reads, position)  # (B, 1 + K, D)
        return checkpoint(
            self.blocks, keys, reads, position, use_reentrant=False
        )  # (B, 1 + K, D)
