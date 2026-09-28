"""Writing a sensor's hidden patches back out of the cells around them."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

# B = batch, Q = cells, K = target patches, S = gathered cells, D = token channels,
# P = patch dimensions.

RING = 1


class Decoder(nn.Module):
    """Gather the cells around each patch by their offsets, and write it from them.

    Attributes:
        shape: The shape of one patch this decoder predicts.
        size: How far a cell runs east, north and in delay rows. (3)
        around: The offset of every gathered cell from the patch's own cell. (S, 3)
        expand: From the shared width up to the decoder width, at unit scale.
        write: From the gathered cells to every sample of the patch.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        shared: int,
        dim: int,
        cell_m: float,
        delay_rows: int,
    ) -> None:
        """Build the decoder for one instrument.

        Args:
            shape: The shape of one patch of the instrument.
            shared: The width the cross-sensor encoder hands tokens at.
            dim: The decoder's width for each gathered cell.
            cell_m: How far a cell runs along either ground axis, in metres.
            delay_rows: How many radar delay rows a cell spans.
        """
        super().__init__()
        self.shape = shape
        self.register_buffer("size", torch.tensor([cell_m, cell_m, float(delay_rows)]))
        steps = torch.arange(-RING, RING + 1)
        around = torch.stack(torch.meshgrid(steps, steps, steps, indexing="ij"), -1)
        self.register_buffer("around", around.reshape(-1, 3))  # (S, 3)
        self.expand = nn.Sequential(nn.Linear(shared, dim), nn.LayerNorm(dim))
        self.write = nn.Sequential(
            nn.Linear(len(self.around) * dim + 3, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, math.prod(shape)),
        )

    def forward(
        self, context: Tensor, offset: Tensor, occupied: Tensor, position: Tensor
    ) -> Tensor:
        """Return the predicted values of every patch asked for.

        Args:
            context: The cell vectors the prediction reads, of any sensor. (B, Q, D)
            offset: The east, north and delay cell of each. (B, Q, 3)
            occupied: Which of them hold anything. (B, Q)
            position: Where each patch asked for sits and reaches, in metres. (B, K, 6)

        Returns:
            prediction: One patch per slot, meaningful where the patch is hidden.
                (B, K, *P)
        """
        batch, cells = occupied.shape
        if position.shape[1] == 0 or cells == 0:
            return position.new_zeros(*position.shape[:2], *self.shape)  # (B, K, *P)
        low = offset.amin(dim=(0, 1)) - RING  # (3)
        extent = offset.amax(dim=(0, 1)) + RING - low + 1  # (3)
        # Each cell of the batch's volume points at its slot, an empty one past the last
        slot = offset.new_full((batch, *extent.tolist()), cells)
        held, at = occupied.nonzero(as_tuple=True)
        placed = offset[held, at] - low  # (N, 3)
        slot[held, placed[:, 0], placed[:, 1], placed[:, 2]] = at
        where = position[..., :3] / self.size  # (B, K, 3)
        home = where.floor().long()  # (B, K, 3)
        # Padding slots sit anywhere, so every look-up is held inside the volume
        near = (home.unsqueeze(2) + self.around - low).clamp(min=0)
        near = torch.minimum(near, extent - 1)  # (B, K, S, 3)
        rows = torch.arange(batch, device=offset.device)[:, None, None]  # (B, 1, 1)
        gathered = slot[rows, near[..., 0], near[..., 1], near[..., 2]]  # (B, K, S)
        read = self.expand(context)  # (B, Q, D')
        read = torch.cat([read, read.new_zeros(batch, 1, read.shape[-1])], dim=1)
        read = read[rows, gathered]  # (B, K, S, D')
        inside = where - home - 0.5  # (B, K, 3)
        written = self.write(torch.cat([read.flatten(2), inside.to(read.dtype)], -1))
        return written.unflatten(-1, self.shape)  # (B, K, *P)
