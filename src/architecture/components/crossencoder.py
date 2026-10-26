"""Bringing the sensors' tokens into one space, and one token that reads them all."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional

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
            position: Each token's patch centre, in metres or rows. (B, K, 3)

        Returns:
            cls: The token read over every instrument together. (B, D)
            tokens: Each instrument's mapped tokens, meaningful where visible.
                (B, K, D)
        """
        sizes = [one.shape[1] for one in keys]
        counted = torch.cat(visible, 1)  # (B, S)
        owner = torch.repeat_interleave(
            torch.tensor(sizes, device=counted.device)
        )  # (S)
        # Each instrument's CLS leads its tokens, reading them and never read back
        lead = functional.pad(owner[1:] != owner[:-1], (1, 0), value=True)  # (S)
        apart = (owner[:, None] == owner) & (lead[:, None] | ~lead)  # (S, S)
        reads = torch.cat(
            [counted.unsqueeze(1), apart & counted.unsqueeze(1)], 1
        )  # (B, 1 + S, S)
        mapped = self.blocks(
            torch.cat([self.cls.expand(len(counted), 1, -1), *keys], 1),
            functional.pad(reads, (1, 0)),  # (B, 1 + S, 1 + S)
            functional.pad(torch.cat(position, 1), (0, 0, 1, 0)),
        )  # (B, 1 + S, D)
        return mapped[:, 0], list(mapped[:, 1:].split(sizes, 1))
