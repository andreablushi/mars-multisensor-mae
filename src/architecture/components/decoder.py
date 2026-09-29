"""Writing a sensor's hidden patches back out of the cells around them."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from architecture.components.acquisition_encoding import AcquisitionEncoding
from architecture.components.positional_encoding import PositionalEncoding
from architecture.grid import covering_indices, overlapping_boxes

# B = batch, C = cells, K = target patches, N = hidden patches, M = window cells,
# D = token channels, P = patch dimensions.

PATCHES = 1024

REACH = 3.0


class Decoder(nn.Module):
    """Let each hidden patch attend over the cells around it, then write it.

    A patch reads the occupied cells within one patch of its own size on every
    side, each placed by where it sits against the patch.

    Attributes:
        shape: The shape of one patch this decoder predicts.
        expand: From the shared width up to the decoder width, at unit scale.
        mask: The token standing in for a hidden patch. (D')
        place: The positional encoding of a cell's offset from the patch.
        acquire: The acquisition encoding of the patch asked for.
        blocks: The transformer, from the patch to its cells.
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
        self.expand = nn.Sequential(nn.Linear(shared, dim), nn.LayerNorm(dim))
        self.mask = nn.Parameter(torch.zeros(dim))  # (D')
        nn.init.normal_(self.mask, std=0.02)
        self.place = PositionalEncoding(dim, stride)
        self.acquire = AcquisitionEncoding(dim)
        block = nn.TransformerDecoderLayer(
            dim,
            heads,
            4 * dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerDecoder(block, depth, norm=nn.LayerNorm(dim))
        self.predict = nn.Linear(dim, math.prod(shape))
        nn.init.zeros_(self.predict.weight)
        nn.init.zeros_(self.predict.bias)

    def forward(
        self,
        context: Tensor,
        context_position: Tensor,
        context_visible: Tensor,
        position: Tensor,
        acquisition: Tensor,
        hidden: Tensor,
    ) -> Tensor:
        """Return the predicted values of the hidden patches.

        Args:
            context: The cells the prediction reads, of any sensor. (B, C, D)
            context_position: Where each sits and reaches, in metres. (B, C, 6)
            context_visible: Which of them hold anything. (B, C)
            position: Where each patch asked for sits and reaches, in metres. (B, K, 6)
            acquisition: How each was taken, nan where unknown. (B, K, 7)
            hidden: Which of them to predict. (B, K)

        Returns:
            prediction: One patch per slot, meaningful where hidden. (B, K, *P)
        """
        prediction = position.new_zeros(*position.shape[:2], *self.shape)
        batch, slot = hidden.nonzero(as_tuple=True)  # (N,), (N,)
        if not len(batch):
            return prediction  # (B, K, *P)
        cells = self.expand(context)  # (B, C, D')
        asked = position[batch, slot]  # (N, 6)
        # A patch's window is its own box, every span tripled
        window = torch.cat([asked[:, :3], asked[:, 3:] * REACH], dim=-1)  # (N, 6)
        told = self.mask + self.acquire(acquisition[batch, slot])  # (N, D')
        written = []
        for part, placed, query in zip(
            batch.split(PATCHES),
            window.split(PATCHES),
            told.split(PATCHES),
            strict=True,
        ):
            near = (
                overlapping_boxes(placed[:, None], context_position[part], 3)
                & context_visible[part]
            )  # (n, C)
            at, chosen, ignored = covering_indices(near, cells.dtype)  # (n, M)
            around = context_position[part[:, None], at]  # (n, M, 6)
            offset = torch.cat(
                [around[..., :3] - placed[:, None, :3], around[..., 3:]], dim=-1
            )  # (n, M, 6)
            read = cells[part[:, None], at] + self.place(offset)  # (n, M, D')
            # A window holding no cell reads zeros, so it writes from its query alone
            read = read * chosen.unsqueeze(-1)  # (n, M, D')
            decoded = self.blocks(
                query.unsqueeze(1), read, memory_key_padding_mask=ignored
            )  # (n, 1, D')
            written.append(self.predict(decoded[:, 0]))  # (n, prod P)
        prediction[batch, slot] = (
            torch.cat(written).unflatten(-1, self.shape).to(prediction.dtype)
        )
        return prediction  # (B, K, *P)
