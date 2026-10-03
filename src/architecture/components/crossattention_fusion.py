"""Reading cells from the sensor patches covering them."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional

from architecture.components.neighbourhood import neighbourhoods
from architecture.components.positional_encoding import PositionalEncoding
from architecture.grid import Cells, TileGrid
from architecture.tokens import Tokens

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
        # One learned query shared by every cell, before its position is added
        self.query = nn.Parameter(torch.zeros(dim))  # (D)
        # Start the query small, as transformer embeddings are
        nn.init.normal_(self.query, std=0.02)
        # Encode where a cell sits and how far it reaches, at the cell's own size
        self.place = PositionalEncoding(dim, cell_m)
        # The attention a cell reads its covering patches with
        self.attend = nn.MultiheadAttention(dim, heads, batch_first=True)
        # Normalise what a cell read before it is made unit length
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        tokens: dict[str, Tensor],
        batch: dict[str, Tokens],
        counted: dict[str, Tensor],
        cells: Cells,
        read: Sequence[str],
    ) -> TileGrid:
        """Return the grid one set of instruments makes of each tile.

        Args:
            tokens: Each instrument's tokens from the cross-sensor encoder. (B, K, D)
            batch: Each instrument's patches over the batch, which place the tokens.
            counted: Which of its tokens count: visible while training, else present.
            cells: The cells the batch's patches reach.
            read: Which instruments this grid is built from.

        Returns:
            grid: One unit vector per usable cell.
        """
        # Where each cell sits and how far it reaches
        placed = cells.position  # (B, Q, 6)
        # The tokens of every instrument read, side by side
        held = torch.cat([tokens[name] for name in read], dim=1)  # (B, S, D)
        # Where each of those tokens' patches sits and reaches
        where = torch.cat([batch[name].position for name in read], dim=1)  # (B, S, 6)
        # Which of those tokens may be read
        taken = torch.cat([counted[name] for name in read], dim=1)  # (B, S)
        # With no token at all, every cell is left empty
        if held.shape[1] == 0:
            return TileGrid(
                held.new_zeros(*placed.shape[:2], held.shape[-1]),
                torch.zeros_like(cells.present),
                cells.offset,
            )
        # Every cell of every tile as one query
        boxes = placed.flatten(0, 1)  # (B * Q, 6)
        # The tile each query belongs to
        tiles = torch.arange(len(placed), device=placed.device).repeat_interleave(
            placed.shape[1]
        )  # (B * Q,)
        attended, covered = [], []
        # Read the cells a chunk at a time, so a tile's cells fit in memory
        for rows, at, chosen, ignored in neighbourhoods(
            boxes, tiles, where, taken, CELLS
        ):
            # The tokens covering each cell of the chunk, padded to one count
            picked = held[tiles[rows, None], at]  # (n, M, D)
            # Each cell asks with the shared query, placed where it sits
            asked = self.query + self.place(boxes[rows])  # (n, D)
            # Each cell attends over the tokens covering it, padding skipped
            read_out, _ = self.attend(
                asked.unsqueeze(1),
                picked,
                picked,
                key_padding_mask=ignored,
                need_weights=False,
            )  # (n, 1, D)
            # Keep what each cell read
            attended.append(read_out[:, 0])  # (n, D)
            # Keep whether any token covered it
            covered.append(chosen.any(dim=-1))  # (n,)
        # A cell is occupied when it is a real cell that some token covered
        occupied = cells.present & torch.cat(covered).view(placed.shape[:2])  # (B, Q)
        # Normalise what each cell read and make it unit length
        values = functional.normalize(
            self.norm(torch.cat(attended).view(*placed.shape[:2], -1)), dim=-1
        )  # (B, Q, D)
        # Empty cells are zeroed, so only occupied ones carry a vector
        return TileGrid(values * occupied.unsqueeze(-1), occupied, cells.offset)
