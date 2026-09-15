"""Writing a sensor's hidden patches back out of tokens, whichever sensor read them."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from architecture.components.modality_encoding import ModalityEncoding
from architecture.components.positional_encoding import PositionalEncoding
from architecture.components.transformer import Transformer
from dataset.patches import channel_axis


class Decoder(nn.Module):
    """Attend over the tokens read and one stand-in per patch asked for, then write it.

    Attributes:
        shape: The shape of one patch this decoder predicts.
        at: Which axis the channels run along, or None where a patch holds one.
        ground: The shape of one channel of it.
        expand: From the shared width up to the decoder width.
        mask: The token standing in for a hidden patch. (D')
        place: The positional encoding.
        blocks: The transformer.
        modality: What the instrument is and what each channel of it measures.
        predict: From a channel's token to its ground samples, shared by all.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        axes: tuple[str, ...],
        shared: int,
        dim: int,
        heads: int,
        depth: int,
        stride: float,
    ) -> None:
        """Build the decoder for one instrument.

        Args:
            shape: The shape of one patch of the instrument.
            axes: What each of its axes holds, in that same order.
            shared: The width the cross-sensor encoder hands tokens at.
            dim: The decoder's token width.
            heads: How many attention heads each block runs.
            depth: How many blocks are stacked.
            stride: How far apart two neighbouring patch centres sit, in metres.
        """
        super().__init__()
        self.shape = shape
        self.at = channel_axis(axes)
        # Isolate spatial dimensions excluding the channel axis
        self.ground = tuple(size for at, size in enumerate(shape) if at != self.at)
        self.modality = ModalityEncoding(axes, dim)
        # Project the cross-sensor encoder width up to the decoder width
        self.expand = nn.Linear(shared, dim)
        # Initialize learnable mask token used as a placeholder for hidden patches
        self.mask = nn.Parameter(torch.zeros(dim))  # (D')
        nn.init.normal_(self.mask, std=0.02)
        # Continuous geospatial positional encodings
        self.place = PositionalEncoding(dim, stride)
        # Decoder Transformer stack for cross-token self-attention
        self.blocks = Transformer(dim, heads, depth)
        # Map a token back to one channel's flattened ground samples
        self.predict = nn.Linear(dim, math.prod(self.ground))

    def forward(
        self,
        context: Tensor,
        context_position: Tensor,
        context_visible: Tensor,
        position: Tensor,
        channels: Tensor,
        hidden: Tensor,
    ) -> Tensor:
        """Return the predicted values of the hidden patches.

        Args:
            context: The shared tokens the prediction reads, of any sensor. (B, C, D)
            context_position: Where each sits and reaches, in metres. (B, C, 6)
            context_visible: Which of them the encoder read. (B, C)
            position: Where each patch asked for sits and reaches, in metres. (B, K, 6)
            channels: What each channel of each of those patches measures. (B, K, C)
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
        # Slice reconstructed query tokens and add sensor modality/channel metadata
        held = decoded[:, read.shape[1] :].unsqueeze(2) + self.modality(
            channels
        )  # (B, K, C, D')
        # Write the tokens out as samples and unflatten to the patch shape
        spread = self.predict(held).unflatten(-1, self.ground)  # (B, K, C, *ground)
        # Move the channel axis back where the patch holds it
        if self.at is None:
            return spread.squeeze(2)  # (B, K, *P)
        return spread.movedim(2, 2 + self.at)  # (B, K, *P)
