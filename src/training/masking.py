"""Hiding patches from their encoder, as the cross-sensor correspondences do."""

from __future__ import annotations

from dataclasses import replace

import torch

from architecture.mae import CrossSensorMAE, Reconstruction
from architecture.models import Cells, Tokens


def random_correspondence(
    batch: dict[str, Tokens], ratio: float, generator: torch.Generator
) -> dict[str, Tokens]:
    """Return the batch with each instrument's patches hidden on a draw of its own.

    Args:
        batch: Each instrument's patches over the batch, keyed as ODE names it.
        ratio: The share of a sensor's present patches to hide, rounded down.
        generator: What fixes the draw, on the batch's own device.

    Returns:
        masked: The same patches, `visible` now what each encoder may read.
    """
    masked = {}
    for name, tokens in batch.items():
        present = tokens.present  # (B, K)
        noise = torch.rand(present.shape, generator=generator, device=present.device)
        # Padding is given the highest noise, so only present patches are hidden.
        order = noise.masked_fill(~present, 2.0).argsort(dim=1)  # (B, K)
        rank = order.argsort(dim=1)  # (B, K)
        hidden = rank < (ratio * present.sum(dim=1, keepdim=True)).floor()  # (B, K)
        masked[name] = replace(tokens, visible=present & ~hidden)
    return masked


def masked_reconstruction(
    model: CrossSensorMAE,
    batch: dict[str, Tokens],
    cells: Cells,
    mask_ratio: float,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[dict[str, Tokens], Reconstruction]:
    """Return one batch masked on the device, and what the model rebuilt of it.

    Args:
        model: The model, on the device.
        batch: Each instrument's patches over the batch.
        cells: The cells the batch's patches reach.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        generator: What fixes the mask, on the device.
        device: Where the model runs.

    Returns:
        masked: The patches, on the device, `visible` what each encoder may read.
        reconstruction: What the model predicted from them.
    """
    # Transfer input tensors to execution device
    batch = {name: tokens.to(device) for name, tokens in batch.items()}
    # Apply random sensor masking
    masked = random_correspondence(batch, mask_ratio, generator)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        return masked, model(masked, cells.to(device))
