"""What one stage hands the next: a sensor's patches, and the cells they reach."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from architecture.grid import Cells, tile_cells

# B = batch, K = patches, P = patch dimensions.


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
        return Tokens(
            self.values.to(device),
            self.valid.to(device),
            self.position.to(device),
            self.visible.to(device),
            self.present.to(device),
        )


def token_batch_padding(
    samples: list[tuple[dict[str, dict[str, np.ndarray]], str, np.ndarray | None]],
    cell_m: float,
    delay_rows: int,
    full_grid: bool = False,
) -> tuple[dict[str, Tokens], Cells, list[str]]:
    """Return one batch of every instrument's tokens, the cells they reach, and whose.

    Args:
        samples: Each tile's patch arrays, identity and optional ground bounds.
        cell_m: How far a cell runs along the ground, in metres.
        delay_rows: How many radar delay rows a cell spans.
        full_grid: Whether to fill each tile's full volume at evaluation.

    Returns:
        batch: Each instrument's patches over the batch, keyed as ODE names it.
        cells: Sparse training cells or full evaluation volumes.
        identities: The tile each read belongs to, in the batch's own order.
    """
    batch = {}
    for name in samples[0][0]:
        held = [sample[name] for sample, _, _ in samples]
        counts = torch.tensor([len(one["values"]) for one in held])  # (B,)
        slots = torch.arange(int(counts.max()))  # (K,)
        padded = {
            key: pad_sequence(
                [torch.as_tensor(one[key]) for one in held], batch_first=True
            )
            for key in held[0]
        }
        present = slots.unsqueeze(0) < counts.unsqueeze(1)  # (B, K)
        batch[name] = Tokens(**padded, visible=present, present=present)
    return (
        batch,
        tile_cells(samples, cell_m, delay_rows, full_grid),
        [identity for _, identity, _ in samples],
    )
