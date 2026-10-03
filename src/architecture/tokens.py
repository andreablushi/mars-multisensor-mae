"""What one stage hands the next: a sensor's patches, and the cells they reach."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from architecture.grid import Cells

# B = batch, K = patches, P = patch dimensions.


class Tokens(NamedTuple):
    """One sensor's patches over a batch of tiles, padded to one count.

    Attributes:
        values: The normalised patches, zero where padded. (B, K, *P)
        measured: Whether each sample is a measurement, false over padding. (B, K, *P')
        position: The patch centre and its span, in metres. (B, K, 6)
    """

    values: Tensor
    measured: Tensor
    position: Tensor


def token_batch_padding(
    samples: list[tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray], str]],
) -> tuple[dict[str, Tokens], Cells, list[str]]:
    """Return one batch of every instrument's tokens, the cells they reach, and whose.

    Args:
        samples: Each tile's patch arrays, the cells they reach, and its identity.

    Returns:
        batch: Each instrument's patches over the batch, keyed as ODE names it.
        cells: The cells the batch's patches reach.
        identities: The tile each read belongs to, in the batch's own order.
    """
    batch = {}
    # Pad each instrument's patches across the batch's tiles
    for name in samples[0][0]:
        # This instrument's arrays in every tile
        held = [sample[name] for sample, _, _ in samples]
        # Every array padded with zeros to the most patches any tile has
        padded = {
            key: pad_sequence(
                [torch.as_tensor(one[key]) for one in held], batch_first=True
            )
            for key in held[0]
        }
        # Values in float32, whatever the stored type
        padded["values"] = padded["values"].float()
        # The instrument's patches over the batch
        batch[name] = Tokens(**padded)
    # The cells of every tile
    reached = [cells for _, cells, _ in samples]
    # How many cells each tile has
    counts = torch.tensor([len(one["offset"]) for one in reached])  # (B,)
    # The cell slots of the tile with the most
    slots = torch.arange(int(counts.max()))  # (Q,)
    # Offsets and positions padded with zeros, and which slots are real cells
    cells = Cells(
        *(
            pad_sequence(
                [torch.as_tensor(one[key]) for one in reached], batch_first=True
            )
            for key in ("offset", "position")
        ),
        slots.unsqueeze(0) < counts.unsqueeze(1),
    )
    # The patches, the cells and the tile names, in the batch's order
    return batch, cells, [identity for _, _, identity in samples]
