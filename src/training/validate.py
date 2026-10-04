"""Measuring a model over one split."""

from __future__ import annotations

from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from architecture.mae import CrossSensorMAE
from training.loss import loss_terms
from training.step import batch_errors, device_batch, drawn_masks


def validation_terms(
    model: CrossSensorMAE,
    loader: DataLoader,
    mask_ratio: float,
    drop_ratio: float,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    """Return the loss terms over one split.

    Args:
        model: The model, which is switched to evaluation.
        loader: The split, in batches.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        drop_ratio: The chance each instrument is left out of the grid.
        seed: What fixes the masks.
        device: Where the model runs.

    Returns:
        metrics: Every loss term over the whole split, as if it were one batch.
    """
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    sums, counts = defaultdict(float), defaultdict(float)
    with torch.no_grad():
        for batch, cells, _ in loader:
            batch, cells = device_batch(batch, cells, device)
            visible, hidden, kept, scored = drawn_masks(
                batch, mask_ratio, drop_ratio, generator
            )
            for term, patches in scored.items():
                counts[term] += float(patches.sum())
            held = batch_errors(model, batch, cells, visible, hidden, kept, scored)
            for term, value in held.items():
                sums[term] += value
    return {name: float(value) for name, value in loss_terms(sums, counts).items()}
