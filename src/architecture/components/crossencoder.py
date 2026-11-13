"""Bringing the sensors' tokens into one space, and one token that reads them all."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional

from architecture.components.transformer import Transformer


class CrossSensorEncoder(nn.Module):
    """Attend over each sensor's tokens on its own, on weights every sensor shares.

    One CLS token reads every sensor's tokens, and none of them reads it back, so
    each sensor is attended apart and the CLS as one query over them all.

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
        cls = self.cls.expand(len(keys[0]), 1, -1)  # (B, 1, D)
        width = sum(one.shape[1] for one in keys)
        # The CLS is the one query over every instrument's tokens, itself the last key
        every = functional.pad(torch.cat(visible, 1), (0, 1), value=True)  # (B, S + 1)
        placed = functional.pad(torch.cat(position, 1), (0, 0, 0, 1))  # (B, S + 1, 3)
        for block in self.blocks.blocks:
            read = block(
                torch.cat([*keys, cls], 1), every[:, None, None], placed, width
            )
            mapped = []
            for held, counted, where in zip(keys, visible, position, strict=True):
                count = held.shape[1]
                # Each instrument's CLS leads its tokens, reading them and never read
                heard = torch.ones(count, count, dtype=torch.bool, device=held.device)
                heard[1:, 0] = False
                # A token always reads itself, so a row with no other key stays finite
                itself = torch.eye(count, dtype=torch.bool, device=held.device)
                reads = counted.unsqueeze(1) & heard | itself  # (B, K, K)
                mapped.append(block(held, reads.unsqueeze(1), where, 0))
            keys, cls = mapped, read[:, width:]
        return self.blocks.norm(cls)[:, 0], [self.blocks.norm(held) for held in keys]
