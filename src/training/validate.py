"""Measuring a model over one split."""

from __future__ import annotations

from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from architecture.mae import CrossSensorMAE
from training.loss import csmae_loss
from training.masking import masked_reconstruction


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
        metrics: Every loss term averaged over the batches, on the same mask.
    """
    model.eval()  # Switch model to evaluation mode (disables dropout/batchnorm updates)
    # Seed generator for reproducible evaluation masks
    generator = torch.Generator(device=device).manual_seed(seed)
    totals = defaultdict(float)  # Accumulate loss components across batches
    batches = 0
    # Disable gradient calculation to save memory and speed up processing
    with torch.no_grad():
        for batch, cells, _ in loader:
            batch, reconstruction = masked_reconstruction(
                model, batch, cells, mask_ratio, generator, device
            )
            # Compute loss metrics on the masked batch
            terms = csmae_loss(reconstruction, batch)
            # Sum each individual loss term for batch averaging later
            for name, value in terms.items():
                totals[name] += float(value)
            batches += 1
    # Return averaged dictionary of all evaluation loss components
    return {name: value / max(batches, 1) for name, value in totals.items()}
