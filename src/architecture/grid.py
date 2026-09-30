"""Cells of a tile and the features read into them."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import torch
from building.configs import sharad
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

TOUCH = 1e-2

# B = batch, Q = cells, D = feature channels.


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
            cells: Every tensor moved there, copied beside the work when pinned.
        """
        return Cells(
            *(
                getattr(self, one.name).to(device, non_blocking=True)
                for one in fields(self)
            )
        )

    def pin_memory(self) -> Cells:
        """Return the same cells in page-locked memory, which the loader copies from.

        Returns:
            cells: Every tensor pinned.
        """
        return Cells(*(getattr(self, one.name).pin_memory() for one in fields(self)))


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


def overlapping_boxes(first: Tensor, second: Tensor, axes: int) -> Tensor:
    """Return whether two sets of centred boxes overlap on each of their first axes.

    Args:
        first: Where each box sits and reaches, broadcastable to the second. (..., 6)
        second: Where each other box sits and reaches. (..., 6)
        axes: How many of east, north and delay the boxes are compared along.

    Returns:
        overlapping: Whether each pair overlaps along every one of those axes.
    """
    apart = (first[..., :axes] - second[..., :axes]).abs()
    first_span, second_span = first[..., 3 : 3 + axes], second[..., 3 : 3 + axes]
    # Only the smaller box is shrunk, so one of no span still lands where it lies
    smaller = torch.minimum(first_span, second_span)
    reach = (first_span + second_span) / 2 - TOUCH * smaller
    return (apart < reach).all(dim=-1)


def covering_indices(
    covering: Tensor, dtype: torch.dtype
) -> tuple[Tensor, Tensor, Tensor]:
    """Return where each row's covering boxes sit, padded to the most any row holds.

    Args:
        covering: Whether each box covers each other box. (..., S)
        dtype: What the rows are ranked as, that of the tokens they then gather.

    Returns:
        at: The index of each covering box, then of padding. (..., M)
        chosen: Whether each index is a covering box rather than padding. (..., M)
        ignored: The padding attention skips, none in a row nothing covers. (..., M)
    """
    count = max(int(covering.sum(dim=-1).max()), 1)
    chosen, at = covering.to(dtype).topk(count, dim=-1)  # (..., M)
    chosen = chosen.bool()  # (..., M)
    return at, chosen, ~chosen & chosen.any(dim=-1, keepdim=True)


def tile_cells(
    samples: list[tuple[dict[str, dict[str, np.ndarray]], str]],
    cell_m: float,
    delay_rows: int,
) -> Cells:
    """Return the cells the patches of each tile in a batch reach.

    Args:
        samples: The patch arrays and identity of each tile.
        cell_m: How far a cell runs along either ground axis, in metres.
        delay_rows: How many radar delay rows a cell spans.

    Returns:
        cells: The tile cells, padded across the batch.
    """
    reached = []
    depth_cells = (sharad.DELAY_ROWS + delay_rows - 1) // delay_rows
    size = np.array([cell_m, cell_m, delay_rows])
    for sample, _ in samples:
        placed = np.concatenate([held["position"] for held in sample.values()])
        reach = placed[:, 3:] / 2 - TOUCH * np.minimum(placed[:, 3:], size)
        low = np.floor((placed[:, :3] - reach) / size)
        high = np.ceil((placed[:, :3] + reach) / size) - 1
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
