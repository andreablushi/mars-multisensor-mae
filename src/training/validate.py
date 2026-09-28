"""Measuring a model over one split."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection

import torch
from torch.utils.data import DataLoader

from architecture.mae import CrossSensorMAE
from training.loss import csmae_loss
from training.step import masked_reconstruction


def validation_terms(
    model: CrossSensorMAE,
    loader: DataLoader,
    mask_ratio: float,
    unnormalised_patches: Collection[str],
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    """Return the loss terms over one split.

    Args:
        model: The model, which is switched to evaluation.
        loader: The split, in batches.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        unnormalised_patches: The instruments whose targets keep their own scale.
        seed: What fixes the masks.
        device: Where the model runs.

    Returns:
        metrics: Every loss term averaged over the batches, on the same mask.
    """
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    totals = defaultdict(float)
    batches = 0
    with torch.no_grad():
        for batch, cells, _ in loader:
            batch, reconstruction = masked_reconstruction(
                model, batch, cells, mask_ratio, generator, device
            )
            terms = csmae_loss(reconstruction, batch, unnormalised_patches)
            for name, value in terms.items():
                totals[name] += float(value)
            batches += 1
    return {name: value / max(batches, 1) for name, value in totals.items()}
