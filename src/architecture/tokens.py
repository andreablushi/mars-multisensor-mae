"""What one stage hands the next: a sensor's patches, and the cells they reach."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from architecture.grid import Cells, tile_cells

# B = batch, K = patches, P = patch dimensions.


class Tokens(NamedTuple):
    """One sensor's patches over a batch of tiles, padded to one count.

    Attributes:
        values: The normalised patches, zero where padded. (B, K, *P)
        measured: Whether each sample is a measurement, false over padding. (B, K, *P')
        position: The patch centre and its span, in metres. (B, K, 6)
        measured_slots: Whether a slot holds a patch rather than padding. (B, K)
    """

    values: Tensor
    measured: Tensor
    position: Tensor
    measured_slots: Tensor


def token_batch_padding(
    samples: list[tuple[dict[str, dict[str, np.ndarray]], str]],
    cell_m: float,
    delay_rows: int,
) -> tuple[dict[str, Tokens], Cells, list[str]]:
    """Return one batch of every instrument's tokens, the cells they reach, and whose.

    Args:
        samples: Each tile's patch arrays and identity.
        cell_m: How far a cell runs along the ground, in metres.
        delay_rows: How many radar delay rows a cell spans.

    Returns:
        batch: Each instrument's patches over the batch, keyed as ODE names it.
        cells: The cells the batch's patches reach.
        identities: The tile each read belongs to, in the batch's own order.
    """
    batch = {}
    for name in samples[0][0]:
        held = [sample[name] for sample, _ in samples]
        counts = torch.tensor([len(one["values"]) for one in held])  # (B,)
        slots = torch.arange(int(counts.max()))  # (K,)
        padded = {
            key: pad_sequence(
                [torch.as_tensor(one[key]) for one in held], batch_first=True
            )
            for key in held[0]
        }
        padded["values"] = padded["values"].float()
        measured_slots = slots.unsqueeze(0) < counts.unsqueeze(1)  # (B, K)
        batch[name] = Tokens(**padded, measured_slots=measured_slots)
    return (
        batch,
        tile_cells(samples, cell_m, delay_rows),
        [identity for _, identity in samples],
    )
