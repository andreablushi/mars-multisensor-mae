"""One masked reconstruction step on the model device, hiding patches at random."""

from __future__ import annotations

import torch
from torch import Tensor

from architecture.grid import Cells
from architecture.mae import CrossSensorMAE
from architecture.tokens import Tokens
from training.loss import csmae_loss


def reconstruction_terms(
    model: CrossSensorMAE,
    batch: dict[str, Tokens],
    cells: Cells,
    mask_ratio: float,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, Tensor]:
    """Return the loss terms of one batch, its patches hidden on a random draw.

    Args:
        model: The model, on the device.
        batch: Each instrument's patches over the batch.
        cells: The cells the batch's patches reach.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        generator: What fixes the mask, on the device.
        device: Where the model runs.

    Returns:
        terms: "umr/<sensor>", "cmr/<sensor>", and "loss".
    """
    batch = {
        name: Tokens(*(one.to(device, non_blocking=True) for one in tokens))
        for name, tokens in batch.items()
    }
    cells = Cells(*(one.to(device, non_blocking=True) for one in cells))
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
    with torch.autocast(device.type, dtype=torch.bfloat16):
        reconstruction = model(batch, visible, hidden, cells)
    return csmae_loss(reconstruction, batch, visible, hidden)
