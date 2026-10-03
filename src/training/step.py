"""Run one masked reconstruction step on the model device, hiding patches at random."""

from __future__ import annotations

import torch
from torch import Tensor

from architecture.grid import Cells
from architecture.mae import CrossSensorMAE, Reconstruction
from architecture.tokens import Tokens


def masked_reconstruction(
    model: CrossSensorMAE,
    batch: dict[str, Tokens],
    cells: Cells,
    mask_ratio: float,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[dict[str, Tokens], dict[str, Tensor], Reconstruction]:
    """Return the patches on the model device, what each encoder read, and the rest.

    Args:
        model: The model, on the device.
        batch: Each instrument's patches over the batch.
        cells: The cells the batch's patches reach.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        generator: What fixes the mask, on the device.
        device: Where the model runs.

    Returns:
        batch: The patches, on the device.
        visible: Which patches each encoder may read, drawn for each instrument on
            its own. (B, K)
        reconstruction: The model's predictions for hidden patches.
    """
    batch = {
        name: Tokens(*(one.to(device, non_blocking=True) for one in tokens))
        for name, tokens in batch.items()
    }
    cells = Cells(*(one.to(device, non_blocking=True) for one in cells))
    visible = {}
    for name, tokens in batch.items():
        present = tokens.measured_slots  # (B, K)
        noise = torch.rand(present.shape, generator=generator, device=present.device)
        # Padding is given the highest noise, so only present patches are hidden.
        order = noise.masked_fill(~present, 2.0).argsort(dim=1)  # (B, K)
        rank = order.argsort(dim=1)  # (B, K)
        hidden = rank < (mask_ratio * present.sum(1, keepdim=True)).floor()  # (B, K)
        visible[name] = present & ~hidden  # (B, K)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        return batch, visible, model(batch, visible, cells)
