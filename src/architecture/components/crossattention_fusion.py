"""Reading volume cells from sensor patches and tile context."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional

from architecture.components.attention import Attention
from architecture.components.span_encoding import SpanEncoding
from architecture.grid import Cells, TileGrid, overlapping_boxes

# B = batch, Q = cells, S = source patches, K = instrument patches, D = token channels.


class CrossAttentionFusion(nn.Module):
    """Read each volume cell from local patches or tile context.

    A cell reads overlapping patches when they exist. Otherwise it reads all
    visible patches of that instrument in the tile.

    Attributes:
        query: What a cell asks with, before it is placed. (D)
        place: The span encoding, at the cell's own size.
        attend: The attention from a cell to visible patches.
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
        self.query = nn.Parameter(torch.zeros(dim))  # (D)
        nn.init.normal_(self.query, std=0.02)
        self.place = SpanEncoding(dim, cell_m)
        self.attend = Attention(dim, heads)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        tokens: dict[str, Tensor],
        position: dict[str, Tensor],
        counted: dict[str, Tensor],
        cells: Cells,
        read: Sequence[str],
    ) -> TileGrid:
        """Return the grid one set of instruments makes of each tile.

        Args:
            tokens: Each instrument's tokens from the cross-sensor encoder. (B, K, D)
            position: Where each of its patches sits and reaches, in metres. (B, K, 6)
            counted: Which of its tokens count: visible while training, else present.
            cells: Sparse training cells or the full evaluation volume.
            read: Which instruments this grid is built from.

        Returns:
            grid: One unit vector per usable cell.
        """
        placed = cells.position  # (B, Q, 6)
        asked = self.query + self.place(placed)  # (B, Q, D)
        held = torch.cat([tokens[name] for name in read], dim=1)  # (B, S, D)
        where = torch.cat([position[name] for name in read], dim=1)  # (B, S, 6)
        taken = torch.cat([counted[name] for name in read], dim=1)  # (B, S)
        if held.shape[1] == 0:
            return TileGrid(
                asked.new_zeros(asked.shape),
                torch.zeros_like(cells.present),
                cells.offset,
                placed,
            )
        reaching = (
            overlapping_boxes(placed[:, :, None], where[:, None], 3) & taken[:, None]
        )  # (B, Q, S)
        occupied = cells.present & taken.any(dim=1, keepdim=True)
        blocked = torch.where(
            reaching.any(dim=-1, keepdim=True), ~reaching, ~taken[:, None]
        )
        blocked[~occupied] = False
        attended = self.attend(asked, placed, held, where, ~blocked)  # (B, Q, D)
        values = functional.normalize(self.norm(attended), dim=-1)  # (B, Q, D)
        return TileGrid(
            values * occupied.unsqueeze(-1), occupied, cells.offset, cells.position
        )
