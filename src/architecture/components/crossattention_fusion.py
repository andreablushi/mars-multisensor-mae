"""Reading one cell out of every patch reaching it, whichever instrument took them."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional

from architecture.components.positional_encoding import PositionalEncoding
from architecture.models import Cells, FeatureGrid


class CrossAttentionFusion(nn.Module):
    """Read each cell out of the patches reaching it, whichever instrument took them.

    A cell asks for what was measured over the ground it covers, and every
    patch whose span reaches that ground answers, so a patch wider than a cell
    speaks for each of the cells it spans and two patches over the same ground
    are heard together without either being told the other is there.

    Attributes:
        cell_m: How far a cell runs along the ground, in metres.
        query: What a cell asks with, before it is placed. (D)
        place: The positional encoding, at the cell's own size.
        attend: The attention from a cell to the patches reaching it.
        norm: What a cell is normalised by before it is read out.
    """

    def __init__(self, dim: int, heads: int, cell_m: float) -> None:
        """Build the fusion for one token width.

        Args:
            dim: The token width, which is also the cell width.
            heads: How many attention heads the fusion runs.
            cell_m: How far a cell runs along the ground, in metres.
        """
        super().__init__()
        self.cell_m = cell_m
        self.query = nn.Parameter(torch.zeros(dim))  # (D)
        nn.init.normal_(self.query, std=0.02)
        self.place = PositionalEncoding(dim, cell_m)
        self.attend = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        tokens: dict[str, Tensor],
        position: dict[str, Tensor],
        counted: dict[str, Tensor],
        cells: Cells,
        read: Sequence[str],
    ) -> FeatureGrid:
        """Return the grid one set of instruments makes of each feature.

        Args:
            tokens: Each instrument's tokens from the cross-sensor encoder. (B, K, D)
            position: Where each of its patches sits and reaches, in metres. (B, K, 6)
            counted: Which of its tokens count: visible while training, else present.
            cells: The cells the batch's patches reach, in one order for all of them.
            read: Which instruments this grid is built from.

        Returns:
            grid: One vector per cell, of unit length where an instrument reaches it.
        """
        centre = (cells.offset + 0.5) * self.cell_m  # (B, Q, 2)
        edge = centre.new_zeros(*centre.shape[:-1], 1)  # (B, Q, 1)
        spans = centre.new_full((*centre.shape[:-1], 2), self.cell_m)  # (B, Q, 2)
        # A cell stands on the ground, spans its own width, and reaches no height.
        placed = torch.cat([centre, edge, spans, edge], dim=-1)  # (B, Q, 6)
        asked = self.query + self.place(placed)  # (B, Q, D)
        held = torch.cat([tokens[name] for name in read], dim=1)  # (B, S, D)
        where = torch.cat([position[name] for name in read], dim=1)  # (B, S, 6)
        taken = torch.cat([counted[name] for name in read], dim=1)  # (B, S)
        # A patch reaches a cell when their extents meet along both ground axes.
        apart = (centre.unsqueeze(2) - where[:, None, :, :2]).abs()  # (B, Q, S, 2)
        reach = (where[:, None, :, 3:5] + self.cell_m) / 2  # (B, 1, S, 2)
        reaching = (apart <= reach).all(dim=-1) & taken.unsqueeze(1)  # (B, Q, S)
        occupied = reaching.any(dim=-1) & cells.present  # (B, Q)
        # A cell no patch reaches attends over all of them, and is then thrown away.
        blocked = ~reaching  # (B, Q, S)
        blocked[~occupied] = False
        attended, _ = self.attend(
            asked,
            held,
            held,
            attn_mask=blocked.repeat_interleave(self.attend.num_heads, dim=0),
            need_weights=False,
        )  # (B, Q, D)
        values = functional.normalize(self.norm(attended), dim=-1)  # (B, Q, D)
        return FeatureGrid(values * occupied.unsqueeze(-1), occupied, cells.offset)
