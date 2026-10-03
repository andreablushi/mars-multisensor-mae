"""Measuring a model over one split."""

from __future__ import annotations

from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from architecture.mae import CrossSensorMAE
from training.step import reconstruction_terms


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
        metrics: Every loss term averaged over the batches it was measured in.
    """
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    totals = defaultdict(float)
    counts = defaultdict(int)
    with torch.no_grad():
        for batch, cells, _ in loader:
            terms = reconstruction_terms(
                model, batch, cells, mask_ratio, generator, device
            )
            for name, value in terms.items():
                if not value.isnan():
                    totals[name] += float(value)
                    counts[name] += 1
    return {name: totals[name] / counts[name] for name in totals}
