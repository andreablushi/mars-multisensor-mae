"""Telling how consistent with ice each labelled point of a tile is, one query each."""

from __future__ import annotations

import numpy as np
import torch
from mars_multisensor_dataset.analysis.ground_truth.swim.sample import grid_offsets
from torch import Tensor, nn
from torch.nn import functional

from architecture.components.transformer import Transformer
from architecture.models.context import Context


class AttentiveProbe(nn.Module):
    """SWIM's grid points asking every token of the tile, as Perceiver IO decodes.

    Attributes:
        query: The token every point asks with, told apart by where it sits. (D)
        points: Where each point sits, east and north in metres, rows from the
            north and points from the west, at no delay. (P * P, 3)
        blocks: The cross-attention blocks the points read the tokens through.
        predict: From a point's token to its value of each target.
    """

    def __init__(self, dim: int, heads: int, depth: int, targets: int) -> None:
        """Build the probe for one token width.

        Args:
            dim: The token width.
            heads: How many attention heads each block runs.
            depth: How many cross-attention blocks are stacked.
            targets: How many targets a point is labelled with.
        """
        super().__init__()
        self.query = nn.Parameter(torch.zeros(dim))  # (D)
        nn.init.normal_(self.query, std=0.02)
        points = torch.as_tensor(np.stack(grid_offsets(), -1), dtype=torch.float32)
        self.register_buffer("points", functional.pad(points, (0, 1)).flatten(0, 1))
        self.blocks = Transformer(dim, heads, depth)
        self.predict = nn.Linear(dim, targets)

    def forward(self, context: Context) -> Tensor:
        """Return every point's prediction of each target.

        Args:
            context: Every instrument's tokens side by side, without their CLS.

        Returns:
            prediction: One per point and target. (B, P * P, Z)
        """
        batch, count = context.tokens.shape[:2]
        queries = self.query.expand(batch, len(self.points), -1)  # (B, P * P, D)
        keys = torch.cat([context.tokens, queries], dim=1)  # (B, S + P * P, D)
        placed = torch.cat(
            [context.position, self.points.expand(batch, -1, -1)], dim=1
        )  # (B, S + P * P, 3)
        readable = functional.pad(context.readable, (0, len(self.points)))
        keys = self.blocks(keys, readable.unsqueeze(1), placed, count)
        return self.predict(keys[:, count:])  # (B, P * P, Z)
