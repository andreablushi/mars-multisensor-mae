"""Measuring a model over one split."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from architecture.mae import CrossSensorMAE
from training.step import step_terms


def validation_terms(
    model: CrossSensorMAE,
    loader: DataLoader,
    mask_ratio: float,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    """Return the loss terms over one split.

    Args:
        model: The model, which is switched to evaluation.
        loader: The split, in batches.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        seed: What fixes the masks.
        device: Where the model runs.

    Returns:
        metrics: Every loss term, averaged over the split's batches.
    """
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        terms = [
            step_terms(model, batch, mask_ratio, generator, device)
            for batch, _ in loader
        ]
    return {
        name: float(torch.stack([one[name] for one in terms]).mean())
        for name in terms[0]
    }
