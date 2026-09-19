"""What one stage hands the next: a sensor's patches, and the cells they reach."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


@dataclass(frozen=True, slots=True)
class Tokens:
    """One sensor's patches over a batch of tiles, padded to one count.

    Attributes:
        values: The normalised patches, zero where padded. (B, K, *P)
        valid: Whether each sample is a measurement. (B, K, *P')
        position: The patch centre and its span, in metres. (B, K, 6)
        visible: Whether a slot holds a patch its encoder may read. (B, K)
        present: Whether each slot holds a patch rather than padding. (B, K)
    """

    values: Tensor
    valid: Tensor
    position: Tensor
    visible: Tensor
    present: Tensor

    def to(self, device: torch.device) -> Tokens:
        """Return the same tokens held on one device.

        Args:
            device: The device to hold them on.

        Returns:
            tokens: Every tensor moved there.
        """
        # Transfer all underlying data and mask tensors to target execution device
        return Tokens(
            self.values.to(device),
            self.valid.to(device),
            self.position.to(device),
            self.visible.to(device),
            self.present.to(device),
        )


@dataclass(frozen=True, slots=True)
class Cells:
    """The cells a batch of tiles is cut into, padded to one count.

    A cell is a square of ground the same size for every tile, so one cell
    offset stands for the same place whichever tile holds it and two
    tiles are compared over the offsets they share. How many cells a tile
    holds is its own, since a patch reaches every cell its span covers.

    Attributes:
        offset: Which cell each slot stands for, east then north. (B, Q, 2)
        present: Whether a slot holds a cell rather than padding. (B, Q)
    """

    offset: Tensor
    present: Tensor

    def to(self, device: torch.device) -> Cells:
        """Return the same cells held on one device.

        Args:
            device: The device to hold them on.

        Returns:
            cells: Every tensor moved there.
        """
        return Cells(self.offset.to(device), self.present.to(device))


@dataclass(frozen=True, slots=True)
class TileGrid:
    """A batch of tiles as a grid of cells, each standing for the ground it covers.

    Attributes:
        values: The cell vectors, of unit length where occupied, else zero. (B, Q, D)
        occupied: Whether an instrument it was built from reaches the cell. (B, Q)
        offset: Which cell each slot stands for, east then north. (B, Q, 2)
    """

    values: Tensor
    occupied: Tensor
    offset: Tensor


def collate(
    samples: list[tuple[dict[str, dict[str, np.ndarray]], str]],
    cell_m: float,
) -> tuple[dict[str, Tokens], Cells, list[str]]:
    """Return one batch of every instrument's tokens, the cells they reach, and whose.

    Args:
        samples: What one read of each tile holds, and the tile it belongs to.
        cell_m: How far a cell runs along the ground, in metres.

    Returns:
        batch: Each instrument's patches over the batch, keyed as ODE names it.
        cells: Every cell those patches reach, in one order for the whole batch.
        identities: The tile each read belongs to, in the batch's own order.
    """
    batch = {}
    # Process each instrument/sensor present in the first dataset sample
    for name in samples[0][0]:
        held = [sample[name] for sample, _ in samples]
        # Count number of active patches per sample to find max sequence length K
        counts = torch.tensor([len(one["values"]) for one in held])  # (B,)
        slots = torch.arange(int(counts.max()))  # (K,)
        # Pad variable-length patch arrays across batch samples to uniform length K
        padded = {
            key: pad_sequence(
                [torch.as_tensor(one[key]) for one in held], batch_first=True
            )
            for key in held[0]
        }
        # Construct boolean mask identifying real patches versus zero-padded slots
        present = slots.unsqueeze(0) < counts.unsqueeze(1)  # (B, K)
        # Instantiate Tokens container (defaulting visible patches to present patches)
        batch[name] = Tokens(**padded, visible=present, present=present)
    reached = []
    for sample, _ in samples:
        placed = np.concatenate(
            [held["position"] for held in sample.values()]
        )  # (K, 6)
        low = np.floor((placed[:, :2] - placed[:, 3:5] / 2) / cell_m)  # (K, 2)
        high = np.floor((placed[:, :2] + placed[:, 3:5] / 2) / cell_m)  # (K, 2)
        # A patch reaches every cell between the two corners its span puts it in.
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
        )  # (Q, 2)
        reached.append(torch.as_tensor(offsets))
    counts = torch.tensor([len(one) for one in reached])  # (B,)
    slots = torch.arange(int(counts.max()))  # (Q,)
    cells = Cells(
        offset=pad_sequence(reached, batch_first=True),  # (B, Q, 2)
        present=slots.unsqueeze(0) < counts.unsqueeze(1),  # (B, Q)
    )
    return batch, cells, [identity for _, identity in samples]
