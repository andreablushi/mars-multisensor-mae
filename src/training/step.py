"""One masked reconstruction pass on the model device, hiding patches at random."""

from __future__ import annotations

import torch
from torch import Tensor

from architecture.grid import Cells
from architecture.mae import CrossSensorMAE
from architecture.tokens import Tokens
from training.loss import reconstruction_sums, term_weights


def device_batch(
    batch: dict[str, Tokens], cells: Cells, device: torch.device
) -> tuple[dict[str, Tokens], Cells]:
    """Return one batch held on the model device.

    Args:
        batch: Each instrument's patches over the batch.
        cells: The cells the batch's patches reach.
        device: Where the model runs.

    Returns:
        batch: The patches, on the device.
        cells: The cells, on the device.
    """
    return (
        {
            name: Tokens(*(one.to(device, non_blocking=True) for one in tokens))
            for name, tokens in batch.items()
        },
        Cells(*(one.to(device, non_blocking=True) for one in cells)),
    )


def drawn_masks(
    batch: dict[str, Tokens], mask_ratio: float, generator: torch.Generator
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[tuple[str, str], float]]:
    """Return which patches each encoder reads, which are hidden, and their counts.

    Args:
        batch: Each instrument's patches over the batch, on the device.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        generator: What fixes the mask, on the device.

    Returns:
        visible: Which patches each encoder reads, drawn for each on its own. (B, K)
        hidden: Which patches are predicted, none of them padding. (B, K)
        counts: How many patches each term counts in this batch.
    """
    visible, hidden = {}, {}
    for name, tokens in batch.items():
        present = tokens.measured.flatten(2).any(dim=-1)  # (B, K)
        noise = torch.rand(present.shape, generator=generator, device=present.device)
        # Padding is given the highest noise, so only present patches are hidden.
        order = noise.masked_fill(~present, 2.0).argsort(dim=1)  # (B, K)
        rank = order.argsort(dim=1)  # (B, K)
        drawn = rank < (mask_ratio * present.sum(1, keepdim=True)).floor()  # (B, K)
        visible[name] = present & ~drawn  # (B, K)
        hidden[name] = present & drawn  # (B, K)
    counts = {
        term: float(weight.sum())
        for term, weight in term_weights(batch, visible, hidden).items()
    }
    return visible, hidden, counts


def masked_sums(
    model: CrossSensorMAE,
    batch: dict[str, Tokens],
    cells: Cells,
    visible: dict[str, Tensor],
    hidden: dict[str, Tensor],
) -> dict[tuple[str, str], Tensor]:
    """Return each term's squared error over one batch, summed over its patches.

    Args:
        model: The model, on the device.
        batch: Each instrument's patches over the batch, on the device.
        cells: The cells the batch's patches reach, on the device.
        visible: Which patches each encoder reads. (B, K)
        hidden: Which patches are predicted. (B, K)

    Returns:
        sums: Per instrument asked and instrument read, the summed error. ()
    """
    with torch.autocast(cells.offset.device.type, dtype=torch.bfloat16):
        reconstruction = model(batch, visible, hidden, cells)
    return reconstruction_sums(
        reconstruction, batch, term_weights(batch, visible, hidden)
    )
