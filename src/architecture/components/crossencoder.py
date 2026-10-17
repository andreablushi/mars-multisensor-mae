"""Bringing the sensors' tokens into one space, and one token that reads them all."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from architecture.components.transformer import Transformer

# B = batch, K = patches, S = tokens of every instrument, D = token channels.


class CrossSensorEncoder(nn.Module):
    """Attend over each sensor's tokens on its own, on weights every sensor shares.

    One CLS token reads every sensor's tokens, and none of them reads it back.

    Attributes:
        cls: The token that reads every instrument together. (D)
        blocks: The transformer.
    """

    def __init__(self, dim: int, heads: int, depth: int) -> None:
        """Build the shared stack.

        Args:
            dim: The token width.
            heads: How many attention heads each block runs.
            depth: How many blocks are stacked.
        """
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(dim))  # (D)
        nn.init.normal_(self.cls, std=0.02)
        self.blocks = Transformer(dim, heads, depth)

    def forward(
        self, keys: list[Tensor], visible: list[Tensor], position: list[Tensor]
    ) -> tuple[Tensor, list[Tensor]]:
        """Return the CLS of every instrument, and each one's tokens in shared space.

        Args:
            keys: Each instrument's encoded tokens, each a key and a query. (B, K, D)
            visible: Which of them carry a patch the encoder read. (B, K)
            position: Each token's patch centre, in metres. (B, K, 2)

        Returns:
            cls: The token read over every instrument together. (B, D)
            tokens: Each instrument's mapped tokens, meaningful where visible.
                (B, K, D)
        """
        sizes = [one.shape[1] for one in keys]
        counted = torch.cat(visible, 1)  # (B, S)
        owner = torch.cat(
            [
                counted.new_full((size,), at, dtype=torch.long)
                for at, size in enumerate(sizes)
            ]
        )  # (S)
        reads = torch.cat(
            [counted.unsqueeze(1), (owner[:, None] == owner) & counted.unsqueeze(1)], 1
        )  # (B, 1 + S, S)
        reads = torch.cat([reads.new_zeros(*reads.shape[:2], 1), reads], 2)
        mapped = self.blocks(
            torch.cat([self.cls.expand(len(counted), 1, -1), *keys], 1),
            reads,  # (B, 1 + S, 1 + S)
            torch.cat([position[0].new_zeros(len(counted), 1, 2), *position], 1),
        )  # (B, 1 + S, D)
        return mapped[:, 0], list(mapped[:, 1:].split(sizes, 1))
