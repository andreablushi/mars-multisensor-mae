"""Telling how consistent with ice each labelled point of a tile is, one query each."""

from __future__ import annotations

import torch
from analysis.ground_truth.swim.sample import GRID_POINTS, GRID_SPACING_KM
from common.maths.physics import METRES_PER_KM
from torch import Tensor, nn
from torch.nn import functional

from architecture.components.transformer import Transformer
from architecture.models.context import Context


def grid_points() -> Tensor:
    """Return where SWIM's grid points sit around a tile's centre.

    Returns:
        points: East and north in metres, rows from the north and points from the
            west, at no delay. (P * P, 3)
    """
    steps = (torch.arange(GRID_POINTS) - (GRID_POINTS - 1) / 2) * (
        GRID_SPACING_KM * METRES_PER_KM
    )
    north, east = torch.meshgrid(steps.flip(0), steps, indexing="ij")
    return torch.stack([east, north, torch.zeros_like(east)], -1).flatten(0, 1)


class Segmenter(nn.Module):
    """SWIM's grid points asking every token of the tile, as Perceiver IO decodes.

    Attributes:
        query: The token every point asks with, told apart by where it sits. (D)
        points: Where each point sits, east and north in metres, rows from the
            north and points from the west, at no delay. (P * P, 3)
        blocks: The cross-attention blocks the points read the tokens through.
        predict: From a point's token to its consistency at each depth.
    """

    def __init__(self, dim: int, heads: int, depth: int, depths: int) -> None:
        """Build the segmenter for one token width.

        Args:
            dim: The token width.
            heads: How many attention heads each block runs.
            depth: How many cross-attention blocks are stacked.
            depths: How many depths a point is labelled at.
        """
        super().__init__()
        self.query = nn.Parameter(torch.zeros(dim))  # (D)
        nn.init.normal_(self.query, std=0.02)
        self.register_buffer("points", grid_points())
        self.blocks = Transformer(dim, heads, depth)
        self.predict = nn.Linear(dim, depths)

    def forward(self, cls: Tensor, context: Context) -> Tensor:
        """Return every point's consistency.

        Args:
            cls: The token read over every instrument together, unread. (B, D)
            context: Every instrument's tokens side by side, without their CLS.

        Returns:
            consistency: One per point and depth. (B, P * P, Z)
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
