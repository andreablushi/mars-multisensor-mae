"""One step on the model device: patches hidden, then predicted."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from architecture.mae import CrossSensorMAE
from architecture.models.tokens import Tokens
from training.loss import cmr, mim, total, umr


def drawn_masks(
    batch: dict[str, Tokens],
    mask_ratio: float,
    generators: Mapping[str, torch.Generator],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Return which patches are read and which are hidden.

    Args:
        batch: Each instrument's patches over the batch, on the device.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        generators: What fixes each instrument's mask, on the device.

    Returns:
        visible: Which patches each encoder reads, drawn for each on its own. (B, K)
        hidden: Which patches are predicted, none of them padding. (B, K)
    """
    visible, hidden = {}, {}
    for name, tokens in batch.items():
        present = tokens.present  # (B, K)
        noise = torch.rand(
            present.shape, generator=generators[name], device=present.device
        )
        # Padding is given the highest noise, so only present patches are hidden.
        order = noise.masked_fill(~present, 2.0).argsort(dim=1)  # (B, K)
        rank = order.argsort(dim=1)  # (B, K)
        drawn = rank < (mask_ratio * present.sum(1, keepdim=True)).floor()  # (B, K)
        visible[name] = present & ~drawn  # (B, K)
        hidden[name] = present & drawn  # (B, K)
    return visible, hidden


def batch_step(
    model: CrossSensorMAE,
    batch: dict[str, Tokens],
    mask_ratio: float,
    aligned: Sequence[str],
    pooling: str,
    temperature: float,
    measure: str,
    generators: Mapping[str, torch.Generator],
    device: torch.device,
) -> dict[str, Tensor]:
    """Return one batch's loss terms, its patches hidden and predicted.

    Args:
        model: The model, on the device.
        batch: Each instrument's patches over the batch, as the loader hands it.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        aligned: The instruments MIM aligns, none for no MIM.
        pooling: What MIM aligns of each instrument, its GAP or its CLS.
        temperature: What MIM divides every cosine by.
        measure: The reconstruction error, MSE or RMSE.
        generators: What fixes each instrument's masks, on the device.
        device: Where the model runs.

    Returns:
        terms: Every loss term over the batch, and the loss.
    """
    batch = {name: tokens.to(device) for name, tokens in batch.items()}
    visible, hidden = drawn_masks(batch, mask_ratio, generators)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        predictions, pooled = model(batch, visible, hidden, pooling)
    return total(
        umr(predictions["umr"], batch, hidden, measure),
        cmr(predictions["cmr"], batch, hidden, measure),
        mim(pooled, aligned, temperature),
    )
