"""Writing a sensor's hidden patches back out of tokens, whichever sensor read them."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from architecture.components.modalityencoder import ModalityEncoder
from architecture.components.positional_encoding import PositionalEncoding
from architecture.components.transformer import Transformer
from dataset.patches import channel_axis


class Decoder(nn.Module):
    """Attend over the tokens read and one stand-in per patch asked for, then write it.

    Attributes:
        shape: The shape of one patch this decoder predicts.
        at: Which axis of it the channels run along, or None for an instrument
            whose patch holds a single channel.
        ground: The shape of one channel of it.
        expand: From the shared width up to the decoder width.
        mask: The token standing in for a hidden patch. (D')
        place: The positional encoding.
        blocks: The transformer.
        modality: What the instrument is and what each channel of it measures.
        predict: From one channel's token to that channel's ground samples, the
            same weights writing every channel.
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
            stride: How far apart two neighbouring patch centres of the
                instrument sit, in metres.
        """
        super().__init__()
        self.shape = shape
        self.at = channel_axis(axes)
        self.ground = tuple(size for at, size in enumerate(shape) if at != self.at)
        self.modality = ModalityEncoder(axes, dim)
        self.expand = nn.Linear(shared, dim)
        self.mask = nn.Parameter(torch.zeros(dim))  # (D')
        nn.init.normal_(self.mask, std=0.02)
        self.place = PositionalEncoding(dim, stride)
        self.blocks = Transformer(dim, heads, depth)
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
            context: The shared tokens the prediction reads, of this or of
                another instrument. (B, C, D)
            context_position: Where each of them sits and how far it reaches,
                in metres. (B, C, 6)
            context_visible: Which of them the encoder read. (B, C)
            position: Where each patch of this instrument sits and how far it
                reaches, in metres. (B, K, 6)
            channels: What each channel of each of those patches measures.
                (B, K, C)
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
        held = decoded[:, read.shape[1] :].unsqueeze(2) + self.modality(
            channels
        )  # (B, K, C, D')
        spread = self.predict(held).unflatten(-1, self.ground)  # (B, K, C, *ground)
        if self.at is None:
            return spread.squeeze(2)  # (B, K, *P)
        return spread.movedim(2, 2 + self.at)  # (B, K, *P)
