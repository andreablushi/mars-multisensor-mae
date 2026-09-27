"""Cells of a tile and the features read into them."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from building.configs import sharad
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

# B = batch, Q = volume cells, D = feature channels.


@dataclass(frozen=True, slots=True)
class Cells:
    """The cells a batch of tiles is cut into, padded to one count.

    Attributes:
        offset: The east, north and delay cell of each slot. (B, Q, 3)
        position: Ground metres and delay rows for each centre and span. (B, Q, 6)
        present: Whether a slot holds a cell rather than padding. (B, Q)
    """

    offset: Tensor
    position: Tensor
    present: Tensor

    def to(self, device: torch.device) -> Cells:
        """Return the same cells held on one device.

        Args:
            device: The device to hold them on.

        Returns:
            cells: Every tensor moved there.
        """
        return Cells(
            self.offset.to(device), self.position.to(device), self.present.to(device)
        )


@dataclass(frozen=True, slots=True)
class TileGrid:
    """A batch of tile cells with features read into them.

    Attributes:
        values: The cell vectors, of unit length where occupied, else zero. (B, Q, D)
        occupied: Whether the cell can read any sensor token in its tile. (B, Q)
        offset: The east, north and delay cell of each slot. (B, Q, 3)
        position: Ground metres and delay rows for each centre and span. (B, Q, 6)
    """

    values: Tensor
    occupied: Tensor
    offset: Tensor
    position: Tensor


def tile_cells(
    samples: list[tuple[dict[str, dict[str, np.ndarray]], str, np.ndarray | None]],
    cell_m: float,
    delay_rows: int,
    full_grid: bool = False,
) -> Cells:
    """Return patch cells or the complete volume of each tile in a batch.

    Args:
        samples: The patch arrays, identity and optional metre bounds of each tile.
        cell_m: How far a cell runs along either ground axis, in metres.
        delay_rows: How many radar delay rows a cell spans.
        full_grid: Whether to fill every cell within each tile's bounds.

    Returns:
        cells: The tile cells, padded across the batch.
    """
    reached = []
    depth_cells = (sharad.DELAY_ROWS + delay_rows - 1) // delay_rows
    size = np.array([cell_m, cell_m, delay_rows])
    for sample, _, bounds in samples:
        if full_grid:
            low = np.floor(bounds[0] / cell_m).astype(np.int64)
            high = np.floor(bounds[1] / cell_m).astype(np.int64)
            corners = [((low[0], low[1], 0), (high[0], high[1], depth_cells - 1))]
        else:
            placed = np.concatenate([held["position"] for held in sample.values()])
            low = np.floor((placed[:, :3] - placed[:, 3:] / 2) / size)
            high = np.floor((placed[:, :3] + placed[:, 3:] / 2) / size)
            corners = zip(low.astype(np.int64), high.astype(np.int64), strict=True)
        spread = [
            np.stack(
                np.meshgrid(
                    np.arange(east, east_end + 1),
                    np.arange(north, north_end + 1),
                    np.arange(max(delay, 0), min(delay_end, depth_cells - 1) + 1),
                    indexing="ij",
                ),
                axis=-1,
            ).reshape(-1, 3)
            for (east, north, delay), (east_end, north_end, delay_end) in corners
        ]
        offsets = (
            np.unique(np.concatenate(spread), axis=0)
            if spread
            else np.zeros((0, 3), np.int64)
        )
        reached.append(torch.as_tensor(offsets))
    counts = torch.tensor([len(one) for one in reached])
    slots = torch.arange(int(counts.max()))
    offset = pad_sequence(reached, batch_first=True)
    centre = (offset[..., :2] + 0.5) * cell_m
    delay_start = offset[..., 2].to(centre.dtype) * delay_rows
    delay_end = (delay_start + delay_rows).clamp(max=sharad.DELAY_ROWS)
    depth = ((delay_start + delay_end) / 2).unsqueeze(-1)
    span = centre.new_full(centre.shape, cell_m)
    position = torch.cat(
        [centre, depth, span, (delay_end - delay_start)[..., None]], dim=-1
    )
    return Cells(offset, position, slots.unsqueeze(0) < counts.unsqueeze(1))
