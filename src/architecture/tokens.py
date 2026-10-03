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
    for name in samples[0][0]:
        held = [sample[name] for sample, _, _ in samples]
        padded = {
            key: pad_sequence(
                [torch.as_tensor(one[key]) for one in held], batch_first=True
            )
            for key in held[0]
        }
        padded["values"] = padded["values"].float()
        batch[name] = Tokens(**padded)
    reached = [cells for _, cells, _ in samples]
    counts = torch.tensor([len(one["offset"]) for one in reached])  # (B,)
    slots = torch.arange(int(counts.max()))  # (Q,)
    cells = Cells(
        *(
            pad_sequence(
                [torch.as_tensor(one[key]) for one in reached], batch_first=True
            )
            for key in ("offset", "position")
        ),
        slots.unsqueeze(0) < counts.unsqueeze(1),
    )
    return batch, cells, [identity for _, _, identity in samples]
