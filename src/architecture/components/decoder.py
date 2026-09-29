"""Writing a sensor's hidden patches back out of tokens, whichever sensor read them."""

from __future__ import annotations

import math

import torch
from building.configs import sharad
from common.maths import physics
from torch import Tensor, nn
from torch.nn import functional

from architecture.components.positional_encoding import PositionalEncoding
from architecture.components.transformer import Transformer

# B = batch, C = cells, K = target patches, D = token channels, P = patch dimensions,
# H = heads, N = tokens.

ALIGN = 8

DELAY_ROW_M = physics.SPEED_OF_LIGHT_M_S * sharad.DELAY_INTERVAL_S / 2


class Decoder(nn.Module):
    """Attend over the tokens read and one stand-in per patch asked for, then write it.

    Each head weighs a token less the further it sits, some sharply and some barely.

    Attributes:
        shape: The shape of one patch this decoder predicts.
        cell_m: How far a cell runs along the ground, in metres.
        slopes: How fast each head's attention falls with distance, per cell. (H)
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
        cell_m: float,
    ) -> None:
        """Build the decoder for one instrument.

        Args:
            shape: The shape of one patch of the instrument.
            shared: The width the cross-sensor encoder hands tokens at.
            dim: The decoder's token width.
            heads: How many attention heads each block runs.
            depth: How many blocks are stacked.
            stride: How far apart two neighbouring patch centres sit, in metres.
            cell_m: How far a cell runs along the ground, in metres.
        """
        super().__init__()
        self.shape = shape
        self.cell_m = cell_m
        self.register_buffer("slopes", torch.logspace(1, -4, heads, base=2))  # (H)
        self.expand = nn.Sequential(nn.Linear(shared, dim), nn.LayerNorm(dim))
        self.mask = nn.Parameter(torch.zeros(dim))  # (D')
        nn.init.normal_(self.mask, std=0.02)
        self.place = PositionalEncoding(dim, stride)
        self.blocks = Transformer(dim, heads, depth)
        self.predict = nn.Linear(dim, math.prod(shape))
        nn.init.zeros_(self.predict.weight)
        nn.init.zeros_(self.predict.bias)

    def locality(self, context_position: Tensor, position: Tensor) -> Tensor:
        """Return how much each head favours each token over each, by their distance.

        Args:
            context_position: Where each token read sits and reaches, in m. (B, C, 6)
            position: Where each patch asked for sits and reaches, in m. (B, K, 6)

        Returns:
            bias: Over the tokens read then those asked, padded to a multiple of
                ALIGN so attention reads it without a copy. (B, H, N, N)
        """
        centres = torch.cat([context_position, position], dim=1)[..., :3]  # (B, N, 3)
        centres = centres * centres.new_tensor([1.0, 1.0, DELAY_ROW_M])  # (B, N, 3)
        centres = functional.pad(centres, (0, 0, 0, -centres.shape[1] % ALIGN))
        apart = torch.cdist(centres, centres) / self.cell_m  # (B, N, N)
        bias = -self.slopes[None, :, None, None] * apart[:, None]  # (B, H, N, N)
        dtype = (
            torch.get_autocast_dtype(bias.device.type)
            if torch.is_autocast_enabled(bias.device.type)
            else bias.dtype
        )
        return bias.to(dtype)

    def forward(
        self,
        context: Tensor,
        context_position: Tensor,
        context_visible: Tensor,
        position: Tensor,
        hidden: Tensor,
        bias: Tensor,
    ) -> Tensor:
        """Return the predicted values of the hidden patches.

        Args:
            context: The shared tokens the prediction reads, of any sensor. (B, C, D)
            context_position: Where each sits and reaches, in metres. (B, C, 6)
            context_visible: Which of them the encoder read. (B, C)
            position: Where each patch asked for sits and reaches, in metres. (B, K, 6)
            hidden: Which of them to predict. (B, K)
            bias: What `locality` gave for these same positions. (B, H, N, N)

        Returns:
            prediction: One patch per slot, meaningful where hidden. (B, K, *P)
        """
        # Guard clause for empty target inputs
        if position.shape[1] == 0:
            return position.new_zeros(position.shape[0], 0, *self.shape)  # (B, 0, *P)
        read = self.expand(context) + self.place(context_position)  # (B, C, D')
        asked = self.mask + self.place(position)  # (B, K, D')
        sequence = torch.cat([read, asked], dim=1)  # (B, C + K, D')
        attended = torch.cat([context_visible, hidden], dim=1)  # (B, C + K)
        extra = bias.shape[-1] - sequence.shape[1]
        sequence = functional.pad(sequence, (0, 0, 0, extra))  # (B, N, D')
        attended = functional.pad(attended, (0, extra))  # (B, N)
        decoded = self.blocks(sequence, attended, bias)  # (B, N, D')
        held = decoded[:, read.shape[1] : read.shape[1] + position.shape[1]]
        return self.predict(held).unflatten(-1, self.shape)
