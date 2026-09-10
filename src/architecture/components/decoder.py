"""One instrument's decoder, predicting its hidden patches from any instrument."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from architecture.components.encoder import key_padding, transformer
from architecture.components.fourier import PositionEncoding


class Decoder(nn.Module):
    """A transformer over read tokens and mask tokens, ending in a map to a patch.

    Attributes:
        shape: The shape of one patch this decoder predicts.
        expand: From the sphere up to the decoder width.
        mask: The token standing in for a hidden patch. (D')
        place: The position encoding.
        blocks: The transformer.
        predict: From a decoded mask token to the values of its patch.
    """

    def __init__(
        self, shape: tuple[int, ...], latent: int, dim: int, heads: int, depth: int
    ) -> None:
        """Build the decoder for one instrument.

        Args:
            shape: The shape of one patch of the instrument.
            latent: The sphere's dimension, which every token read has.
            dim: The decoder's token width.
            heads: How many attention heads each block runs.
            depth: How many blocks are stacked.
        """
        super().__init__()
        self.shape = shape
        self.expand = nn.Linear(latent, dim)
        self.mask = nn.Parameter(torch.zeros(dim))  # (D')
        nn.init.normal_(self.mask, std=0.02)
        self.place = PositionEncoding(dim)
        self.blocks = transformer(dim, heads, depth)
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
            context: The sphere tokens the prediction reads, of this or of
                another instrument. (B, C, L)
            context_position: Where each of them sits, in metres. (B, C, 3)
            context_visible: Which of them the encoder read. (B, C)
            position: Where each patch of this instrument sits, in metres.
                (B, K, 3)
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
        decoded = self.blocks(
            sequence, src_key_padding_mask=key_padding(attended)
        )  # (B, C + K, D')
        flat = self.predict(decoded[:, read.shape[1] :])  # (B, K, prod(P))
        return flat.view(*flat.shape[:2], *self.shape)  # (B, K, *P)
