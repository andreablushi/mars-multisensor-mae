"""Reading cells from the sensor patches covering them."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional

from architecture.components.positional_encoding import PositionalEncoding
from architecture.grid import Cells, TileGrid, covering_indices, overlapping_boxes

# B = batch, Q = cells, S = source patches, M = covering patches, D = token channels.

CELLS = 4096


class CrossAttentionFusion(nn.Module):
    """Read each cell from the patches covering it, and nothing else.

    A cell no counted patch covers is left empty.

    Attributes:
        query: What a cell asks with, before it is placed. (D)
        place: The positional encoding, at the cell's own size.
        attend: The attention from a cell to the patches covering it.
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
    ) -> TileGrid:
        """Return the grid one set of instruments makes of each tile.

        Args:
            tokens: Each instrument's tokens from the cross-sensor encoder. (B, K, D)
            position: Where each of its patches sits and reaches, in metres. (B, K, 6)
            counted: Which of its tokens count: visible while training, else present.
            cells: The cells the batch's patches reach.
            read: Which instruments this grid is built from.

        Returns:
            grid: One unit vector per usable cell.
        """
        placed = cells.position  # (B, Q, 6)
        held = torch.cat([tokens[name] for name in read], dim=1)  # (B, S, D)
        where = torch.cat([position[name] for name in read], dim=1)  # (B, S, 6)
        taken = torch.cat([counted[name] for name in read], dim=1)  # (B, S)
        if held.shape[1] == 0:
            return TileGrid(
                held.new_zeros(*placed.shape[:2], held.shape[-1]),
                torch.zeros_like(cells.present),
                cells.offset,
                placed,
            )
        attended, covered = [], []
        # Cells are read a chunk at a time, so a tile's cells fit in memory
        for part in placed.split(CELLS, dim=1):
            reaching = (
                overlapping_boxes(part[:, :, None], where[:, None], 3) & taken[:, None]
            )  # (B, q, S)
            # A cell nothing covers attends anyway, and is emptied below
            at, chosen, ignored = covering_indices(reaching, held.dtype)  # (B, q, M)
            picked = held.gather(
                1, at.flatten(1).unsqueeze(-1).expand(-1, -1, held.shape[-1])
            ).unflatten(1, at.shape[1:])  # (B, q, M, D)
            asked = self.query + self.place(part)  # (B, q, D)
            read_out, _ = self.attend(
                asked.flatten(0, 1).unsqueeze(1),
                picked.flatten(0, 1),
                picked.flatten(0, 1),
                key_padding_mask=ignored.flatten(0, 1),
                need_weights=False,
            )  # (B * q, 1, D)
            attended.append(read_out.view(*part.shape[:2], -1))  # (B, q, D)
            covered.append(chosen.any(dim=-1))  # (B, q)
        occupied = cells.present & torch.cat(covered, dim=1)  # (B, Q)
        values = functional.normalize(
            self.norm(torch.cat(attended, dim=1)), dim=-1
        )  # (B, Q, D)
        return TileGrid(values * occupied.unsqueeze(-1), occupied, cells.offset, placed)
