"""How far one tile's cells stand from another's, as a set of places rather than one."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from architecture.models import TileGrid


def chamfer_distances(
    grids: Sequence[TileGrid], neighbourhood: int | None = None
) -> Tensor:
    """Return how far every tile stands from every other, over the cells they hold.

    A tile is its occupied cells and nothing else, so two are compared by matching
    each cell of one to the nearest cell of the other and averaging both ways. The
    cost of a match is the cosine distance between two cells, which the fusion's
    unit vectors put in [0, 1], so the distance is already normalised. A cell whose
    neighbourhood holds nothing to match stands a whole mismatch from that tile.

    Args:
        grids: One grid per tile, in the order the distances are wanted.
        neighbourhood: How far, in cells along either axis, a cell may be matched
            from its own offset, or None to match it anywhere in the other tile.

    Returns:
        distances: The distance between every pair of tiles, in [0, 1]. (T, T)

    Raises:
        ValueError: When a tile holds no occupied cell to be measured over.
    """
    counts = torch.tensor([int(one.occupied.sum()) for one in grids])  # (T,)
    if not counts.all():
        raise ValueError("a tile holding no occupied cell cannot be measured")
    device = grids[0].values.device
    width = int(counts.max())
    values = grids[0].values.new_zeros(len(grids), width, grids[0].values.shape[-1])
    offsets = values.new_zeros(len(grids), width, 2)
    for at, grid in enumerate(grids):
        values[at, : counts[at]] = grid.values[grid.occupied]
        offsets[at, : counts[at]] = grid.offset[grid.occupied].to(values.dtype)
    counted = counts.to(device, values.dtype)  # (T,)
    held = torch.arange(width, device=device) < counted.unsqueeze(1)  # (T, W)
    distances = values.new_zeros(len(grids), len(grids))
    for at in range(len(grids)):
        cost = (1.0 - torch.einsum("qd,tpd->tqp", values[at], values)) / 2  # (T, W, W)
        matched = held[at].view(1, -1, 1) & held.unsqueeze(1)  # (T, W, W)
        if neighbourhood is not None:
            apart = (offsets[at].view(1, -1, 1, 2) - offsets.unsqueeze(1)).abs()
            matched = matched & (apart.amax(-1) <= neighbourhood)  # (T, W, W)
        cost = cost.masked_fill(~matched, torch.inf)
        forward = cost.amin(dim=2).nan_to_num(posinf=1.0)  # (T, W)
        backward = cost.amin(dim=1).nan_to_num(posinf=1.0)  # (T, W)
        distances[at] = (
            (forward * held[at]).sum(-1) / counted[at]
            + (backward * held).sum(-1) / counted
        ) / 2
    # Both ways round are the same sum, so only the arithmetic tells them apart.
    distances = (distances + distances.T) / 2
    return distances.fill_diagonal_(0.0)
