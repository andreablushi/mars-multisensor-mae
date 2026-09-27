"""Cells of a tile and the features read into them."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from building.configs import sharad
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


@dataclass(frozen=True, slots=True)
class Cells:
    """The cells a batch of tiles is cut into, padded to one count.

    Attributes:
        offset: Which cell each slot stands for, east then north. (B, Q, 2)
        position: The centre and span of each cell, in metres. (B, Q, 6)
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
        occupied: Whether an instrument reaches the cell. (B, Q)
        offset: Which cell each slot stands for, east then north. (B, Q, 2)
        position: The centre and span of each cell, in metres. (B, Q, 6)
    """

    values: Tensor
    occupied: Tensor
    offset: Tensor
    position: Tensor


def tile_cells(
    samples: list[tuple[dict[str, dict[str, np.ndarray]], str]], cell_m: float
) -> Cells:
    """Return cells reached by patches in each tile of a batch.

    Args:
        samples: The patch arrays and identity of each tile.
        cell_m: How far a cell runs along either ground axis, in metres.

    Returns:
        cells: Every reached cell, padded across the batch.
    """
    reached = []
    for sample, _ in samples:
        placed = np.concatenate([held["position"] for held in sample.values()])
        low = np.floor((placed[:, :2] - placed[:, 3:5] / 2) / cell_m)
        high = np.floor((placed[:, :2] + placed[:, 3:5] / 2) / cell_m)
        spread = [
            np.stack(
                np.meshgrid(
                    np.arange(east, east_end + 1),
                    np.arange(north, north_end + 1),
                    indexing="ij",
                ),
                axis=-1,
            ).reshape(-1, 2)
            for (east, north), (east_end, north_end) in zip(
                low.astype(np.int64), high.astype(np.int64), strict=True
            )
        ]
        offsets = (
            np.unique(np.concatenate(spread), axis=0)
            if spread
            else np.zeros((0, 2), np.int64)
        )
        reached.append(torch.as_tensor(offsets))
    counts = torch.tensor([len(one) for one in reached])
    slots = torch.arange(int(counts.max()))
    offset = pad_sequence(reached, batch_first=True)
    centre = (offset + 0.5) * cell_m
    depth = centre.new_full((*centre.shape[:-1], 1), float(sharad.AREOID_ROW))
    span = centre.new_full(centre.shape, cell_m)
    position = torch.cat([centre, depth, span, torch.zeros_like(depth)], dim=-1)
    return Cells(offset, position, slots.unsqueeze(0) < counts.unsqueeze(1))
