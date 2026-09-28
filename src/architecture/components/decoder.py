"""Writing a sensor's hidden patches back out of tokens, whichever sensor read them."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from architecture.components.positional_encoding import PositionalEncoding
from architecture.components.transformer import Transformer

# B = batch, C = cells, K = target patches, D = token channels, P = patch dimensions.


class Decoder(nn.Module):
    """Attend over the tokens read and one stand-in per patch asked for, then write it.

    Attributes:
        shape: The shape of one patch this decoder predicts.
        expand: From the shared width up to the decoder width, at unit scale.
        mask: The token standing in for a hidden patch. (D')
        place: The positional encoding.
        blocks: The transformer.
        predict: From a token to every sample of its patch.
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
            stride: How far apart two neighbouring patch centres sit, in metres.
        """
        super().__init__()
        self.shape = shape
        # Project the cross-sensor encoder width up to the decoder width
        self.expand = nn.Sequential(nn.Linear(shared, dim), nn.LayerNorm(dim))
        # Initialize learnable mask token used as a placeholder for hidden patches
        self.mask = nn.Parameter(torch.zeros(dim))  # (D')
        nn.init.normal_(self.mask, std=0.02)
        # Continuous geospatial positional encodings
        self.place = PositionalEncoding(dim, stride)
        # Decoder Transformer stack for cross-token self-attention
        self.blocks = Transformer(dim, heads, depth)
        # Map a token back to its patch samples
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
            context: The shared tokens the prediction reads, of any sensor. (B, C, D)
            context_position: Where each sits and reaches, in metres. (B, C, 6)
            context_visible: Which of them the encoder read. (B, C)
            position: Where each patch asked for sits and reaches, in metres. (B, K, 6)
            hidden: Which of them to predict. (B, K)

        Returns:
            prediction: One patch per slot, meaningful where hidden. (B, K, *P)
        """
        # Guard clause for empty target inputs
        if position.shape[1] == 0:
            return position.new_zeros(position.shape[0], 0, *self.shape)  # (B, 0, *P)
        # Project visible encoder context tokens and inject spatial coordinates
        read = self.expand(context) + self.place(context_position)  # (B, C, D')
        # Broadcast mask token across target queries and inject spatial coordinates
        asked = self.mask + self.place(position)  # (B, K, D')
        # Concatenate context tokens and target mask queries into a unified sequence
        sequence = torch.cat([read, asked], dim=1)  # (B, C + K, D')
        attended = torch.cat([context_visible, hidden], dim=1)  # (B, C + K)
        # Run combined sequence through Transformer blocks
        decoded = self.blocks(sequence, attended)  # (B, C + K, D')
        held = decoded[:, read.shape[1] :]
        return self.predict(held).unflatten(-1, self.shape)
