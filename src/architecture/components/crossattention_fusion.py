"""Reading cells from the sensor patches covering them."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional

from architecture.components.positional_encoding import PositionalEncoding
from architecture.grid import Cells, TileGrid, neighbourhoods, relative_boxes
from architecture.tokens import Tokens

# B = batch, Q = cells, C = cells of one tile, S = source patches,
# M = covering patches, D = token channels.


class CrossAttentionFusion(nn.Module):
    """Read each cell from the patches covering it, and nothing else.

    A cell no readable patch covers is left empty.

    Attributes:
        query: What every cell asks with. (D)
        place: The positional encoding of a patch against a cell, at the cell's size.
        attend: The attention from a cell to the patches covering it.
    """

    def __init__(self, dim: int, heads: int, cell_m: float) -> None:
        """Build the fusion for one token width.

        Args:
            dim: The token width, which is also the cell width.
            heads: How many attention heads the fusion runs.
            cell_m: How far a cell runs along the ground, in metres.
        """
        super().__init__()
        # One learned query shared by every cell
        self.query = nn.Parameter(torch.zeros(dim))  # (D)
        # Start the query small, as transformer embeddings are
        nn.init.normal_(self.query, std=0.02)
        # Encode where a patch sits against a cell and how far it reaches
        self.place = PositionalEncoding(dim, cell_m)
        # The attention a cell reads its covering patches with
        self.attend = nn.MultiheadAttention(dim, heads, batch_first=True)

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
        # The tokens of every instrument read, side by side
        sources = torch.cat([tokens[name] for name in read], dim=1)  # (B, S, D)
        # Where each of those tokens' patches sits and reaches
        source_boxes = torch.cat([batch[name].position for name in read], dim=1)
        # Which of those tokens may be read, the hidden ones never
        readable = torch.cat([counted[name] for name in read], dim=1)  # (B, S)
        # Every cell starts empty
        values = sources.new_zeros(*cells.present.shape, sources.shape[-1])  # (B, Q, D)
        occupied = torch.zeros_like(cells.present)  # (B, Q)
        # With no token at all, every cell stays empty
        if sources.shape[1] == 0:
            return TileGrid(values, occupied, cells.offset)
        # One tile at a time, so its cells and tokens fit in memory
        for tile, present in enumerate(cells.present):
            # Where each real cell of the tile sits and reaches
            boxes = cells.position[tile, present]  # (C, 6)
            # The readable tokens covering each cell, padded to one count
            at, chosen, ignored = neighbourhoods(
                boxes, source_boxes[tile], readable[tile]
            )  # (C, M)
            # Each covering patch's centre relative to the cell's, beside its spans
            offset = relative_boxes(source_boxes[tile, at], boxes)  # (C, M, 6)
            # Each covering token placed by where its patch sits against the cell
            covering = sources[tile, at] + self.place(offset)  # (C, M, D)
            # Every cell asks with the same query
            asked = self.query.expand(len(boxes), 1, -1)  # (C, 1, D)
            # Each cell attends over the tokens covering it, padding skipped
            attended, _ = self.attend(
                asked, covering, covering, key_padding_mask=ignored, need_weights=False
            )  # (C, 1, D)
            # A cell is occupied when some readable token covers it
            covered = chosen.any(dim=-1)  # (C,)
            # Unit vectors where covered, zero elsewhere
            values[tile, present] = (
                functional.normalize(attended[:, 0], dim=-1) * covered.unsqueeze(-1)
            ).to(values.dtype)
            occupied[tile, present] = covered
        return TileGrid(values, occupied, cells.offset)
