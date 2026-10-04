"""Cells of a tile and the features read into them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
from building.configs import sharad
from torch import Tensor

TOUCH = 1e-2

# B = batch, Q = cells, D = feature channels.


class Cells(NamedTuple):
    """The cells a batch of tiles is cut into, padded to one count.

    Attributes:
        offset: The east, north and delay cell of each slot. (B, Q, 3)
        position: Ground metres and delay rows for each centre and span. (B, Q, 6)
        present: Whether a slot holds a cell rather than padding. (B, Q)
    """

    offset: Tensor
    position: Tensor
    present: Tensor


@dataclass(frozen=True, slots=True)
class TileGrid:
    """A batch of tile cells with features read into them.

    Attributes:
        values: The cell vectors, of unit length where occupied, else zero. (B, Q, D)
        occupied: Whether the cell can read any sensor token in its tile. (B, Q)
        offset: The east, north and delay cell of each slot. (B, Q, 3)
    """

    values: Tensor
    occupied: Tensor
    offset: Tensor


def overlapping_boxes(first: Tensor, second: Tensor, axes: int) -> Tensor:
    """Return whether two sets of centred boxes overlap on each of their first axes.

    Args:
        first: Where each box sits and reaches, broadcastable to the second. (..., 6)
        second: Where each other box sits and reaches. (..., 6)
        axes: How many of east, north and delay the boxes are compared along.

    Returns:
        overlapping: Whether each pair overlaps along every one of those axes.
    """
    # How far apart the two centres are along each axis compared
    apart = (first[..., :axes] - second[..., :axes]).abs()
    # How far each box reaches along those axes
    first_span, second_span = first[..., 3 : 3 + axes], second[..., 3 : 3 + axes]
    # Only the smaller box is shrunk, so one of no span still lands where it lies
    smaller = torch.minimum(first_span, second_span)
    # Two boxes overlap when their centres are closer than half their spans added
    reach = (first_span + second_span) / 2 - TOUCH * smaller
    # Only an overlap along every axis compared counts
    return (apart < reach).all(dim=-1)


def relative_boxes(boxes: Tensor, origin: Tensor) -> Tensor:
    """Return boxes with their centres taken from an origin's, their spans kept.

    Args:
        boxes: Where each box around an origin sits and how far it reaches. (N, M, 6)
        origin: Where each origin sits and how far it reaches. (N, 6)

    Returns:
        boxes: Each box's centre less its origin's, beside its own spans. (N, M, 6)
    """
    return torch.cat([boxes[..., :3] - origin[:, None, :3], boxes[..., 3:]], dim=-1)


def neighbourhoods(
    boxes: Tensor, keys: Tensor, usable: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Return the usable keys each box overlaps, padded to the most any box holds.

    Args:
        boxes: Where each query sits and how far it reaches. (N, 6)
        keys: Where each key sits and how far it reaches, shared or per box. (S, 6)
        usable: Which keys may be read, shaped as the keys. (S,)

    Returns:
        at: The index of each overlapping key, then of padding. (N, M)
        chosen: Whether each index is a key rather than padding. (N, M)
        ignored: The padding attention skips, none in a row nothing reaches. (N, M)
    """
    # Which usable keys each box overlaps along east, north and delay
    reaching = overlapping_boxes(boxes.unsqueeze(-2), keys, 3) & usable  # (N, S)
    # The most keys any box overlaps, at least one so no row is empty
    count = max(int(reaching.sum(dim=-1).max()), 1)
    # Each box's overlapping keys first, then padding, with their indices
    chosen, at = reaching.to(keys.dtype).topk(count, dim=-1)  # (N, M)
    # Which of those are real keys
    chosen = chosen.bool()  # (N, M)
    return at, chosen, ~chosen & chosen.any(dim=-1, keepdim=True)


def tile_cells(
    placed: np.ndarray, cell_m: float, delay_rows: int
) -> dict[str, np.ndarray]:
    """Return the cells one tile's patches reach.

    Args:
        placed: Where every patch of the tile sits and how far it reaches. (N, 6)
        cell_m: How far a cell runs along either ground axis, in metres.
        delay_rows: How many radar delay rows a cell spans.

    Returns:
        cells: Each cell's "offset", its east, north and delay cell (Q, 3), and its
            "position", ground metres and delay rows for its centre and span. (Q, 6)
    """
    # How many delay cells the radargram window holds, the last one partial
    depth_cells = (sharad.DELAY_ROWS + delay_rows - 1) // delay_rows
    # How far one cell runs along east, north and delay
    size = np.array([cell_m, cell_m, delay_rows])
    # Each patch's half span, shrunk a touch so a patch only touching a cell misses it
    reach = placed[:, 3:] / 2 - TOUCH * np.minimum(placed[:, 3:], size)
    # The first cell each patch reaches along each axis
    low = np.floor((placed[:, :3] - reach) / size)
    # The last cell it reaches
    high = np.ceil((placed[:, :3] + reach) / size) - 1
    # Each patch's first and last cell, as integers
    corners = zip(low.astype(np.int64), high.astype(np.int64), strict=True)
    # Every cell inside each patch's range, delay held to the radargram window
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
    # Each cell any patch reaches, once
    offset = np.unique(np.concatenate(spread), axis=0)  # (Q, 3)
    # The ground centre of each cell, in metres from the tile centre
    centre = (offset[:, :2] + 0.5) * cell_m  # (Q, 2)
    # The first delay row of each cell
    delay_start = offset[:, 2] * delay_rows  # (Q,)
    # Its last row, the window's end at most
    delay_end = np.minimum(delay_start + delay_rows, sharad.DELAY_ROWS)  # (Q,)
    # Each cell's centre east, north and delay, then its spans along each
    position = np.column_stack(
        [
            centre,
            (delay_start + delay_end) / 2,
            np.full_like(centre, cell_m),
            delay_end - delay_start,
        ]
    )  # (Q, 6)
    # Positions in float32, as the patches' own
    return {"offset": offset, "position": position.astype(np.float32)}
