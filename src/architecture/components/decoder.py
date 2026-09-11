"""One instrument's decoder, predicting its hidden patches from any instrument."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from architecture.components.positional_encoding import PositionalEncoding
from architecture.components.transformer import Transformer


class Decoder(nn.Module):
    """A transformer over read tokens and mask tokens, ending in a map to a patch.

    Attributes:
        shape: The shape of one patch this decoder predicts.
        expand: From the shared width up to the decoder width.
        mask: The token standing in for a hidden patch. (D')
        place: The positional encoding.
        blocks: The transformer.
        predict: From a decoded mask token to the values of its patch.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        shared: int,
        dim: int,
        heads: int,
        depth: int,
        stride: float,
    ) -> None:
        """Build the decoder for one instrument.

        Args:
            shape: The shape of one patch of the instrument.
            shared: The width the cross-sensor encoder hands tokens at.
            dim: The decoder's token width.
            heads: How many attention heads each block runs.
            depth: How many blocks are stacked.
            stride: How far apart two neighbouring patch centres of the
                instrument sit, in metres.
        """
        super().__init__()
        self.shape = shape
        self.expand = nn.Linear(shared, dim)
        self.mask = nn.Parameter(torch.zeros(dim))  # (D')
        nn.init.normal_(self.mask, std=0.02)
        self.place = PositionalEncoding(dim, stride)
        self.blocks = Transformer(dim, heads, depth)
        self.predict = nn.Linear(dim, math.prod(shape))

    def forward(
        self,
        context: Tensor,
        context_position: Tensor,
        context_visible: Tensor,
        position: Tensor,
        hidden: Tensor,
    ) -> Tensor:
        """Return the predicted values of the hidden patches.

        Args:
            context: The shared tokens the prediction reads, of this or of
                another instrument. (B, C, D)
            context_position: Where each of them sits and how far it reaches,
                in metres. (B, C, 6)
            context_visible: Which of them the encoder read. (B, C)
            position: Where each patch of this instrument sits and how far it
                reaches, in metres. (B, K, 6)
            hidden: Which of them to predict. (B, K)

        Returns:
            prediction: One patch per slot, meaningful where hidden. (B, K, *P)
        """
        if position.shape[1] == 0:
            return position.new_zeros(position.shape[0], 0, *self.shape)  # (B, 0, *P)
        read = self.expand(context) + self.place(context_position)  # (B, C, D')
        asked = self.mask + self.place(position)  # (B, K, D')
        sequence = torch.cat([read, asked], dim=1)  # (B, C + K, D')
        attended = torch.cat([context_visible, hidden], dim=1)  # (B, C + K)
        decoded = self.blocks(sequence, attended)  # (B, C + K, D')
        flat = self.predict(decoded[:, read.shape[1] :])  # (B, K, prod(P))
        return flat.view(*flat.shape[:2], *self.shape)  # (B, K, *P)
