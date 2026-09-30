"""Run one masked reconstruction step on the model device."""

from __future__ import annotations

import torch

from architecture.grid import Cells
from architecture.mae import CrossSensorMAE, Reconstruction
from architecture.tokens import Tokens
from training.masking import random_correspondence


def masked_reconstruction(
    model: CrossSensorMAE,
    batch: dict[str, Tokens],
    cells: Cells,
    mask_ratio: float,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[dict[str, Tokens], Reconstruction]:
    """Return masked patches and their reconstruction on the model device.

    Args:
        model: The model, on the device.
        batch: Each instrument's patches over the batch.
        cells: The cells the batch's patches reach.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        generator: What fixes the mask, on the device.
        device: Where the model runs.

    Returns:
        masked: The patches, with visible marking those the encoder may read.
        reconstruction: The model's predictions for hidden patches.
    """
    batch = {name: tokens.to(device) for name, tokens in batch.items()}
    masked = random_correspondence(batch, mask_ratio, generator)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        return masked, model(masked, cells.to(device))
