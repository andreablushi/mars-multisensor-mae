"""Bringing the sensors' tokens into one space, and one token that reads them all."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional

from mars_multisensor_mae.architecture.components.transformer import Transformer


class CrossSensorEncoder(nn.Module):
    """Attend over each sensor's tokens on its own, on weights every sensor shares.

    Attributes:
        cls: The one query over every instrument's tokens, which none of them reads
            back. (D)
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
        counts = [one.shape[1] for one in keys]
        # Each instrument's tokens read only their own, its CLS leading them never read
        own = []
        for count in counts:
            heard = torch.ones(count, count, dtype=torch.bool, device=cls.device)
            heard[1:, 0] = False
            own.append(heard)
        # The overall CLS, last, reads every token and none reads it back
        reads = functional.pad(torch.block_diag(*own), (0, 1, 0, 1))  # (S + 1, S + 1)
        reads[-1] = True
        every = functional.pad(torch.cat(visible, 1), (0, 1), value=True)  # (B, S + 1)
        placed = functional.pad(torch.cat(position, 1), (0, 0, 0, 1))  # (B, S + 1, 3)
        tokens = self.blocks(
            torch.cat([*keys, cls], 1), every.unsqueeze(1) & reads, placed
        )  # (B, S + 1, D)
        return tokens[:, -1], list(tokens[:, :-1].split(counts, 1))
