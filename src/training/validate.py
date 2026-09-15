"""Measuring a model over one split."""

from __future__ import annotations

from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from architecture.mae import CrossSensorMAE
from config.schema import Config
from training.loss import csmae_loss
from training.masking import random_correspondence


def validate(
    model: CrossSensorMAE, loader: DataLoader, config: Config, device: torch.device
) -> dict[str, float]:
    """Return the loss terms over one split.

    Args:
        model: The model, which is switched to evaluation.
        loader: The split, in batches.
        config: How the split is masked.
        device: Where the model runs.

    Returns:
        metrics: Every loss term averaged over the batches, on the same mask.
    """
    model.eval()  # Switch model to evaluation mode (disables dropout/batchnorm updates)
    generator = torch.Generator(device=device).manual_seed(
        config.dataset.seed
    )  # Seed generator for reproducible evaluation masks
    totals = defaultdict(float)  # Accumulate loss components across batches
    batches = 0
    with (
        torch.no_grad()
    ):  # Disable gradient calculation to save memory and speed up processing
        for batch, _ in loader:
            # Transfer input batch tensors to execution device
            batch = {name: tokens.to(device) for name, tokens in batch.items()}
            # Apply deterministic sensor masking
            batch = random_correspondence(batch, config.training.mask_ratio, generator)
            # Compute loss metrics on the masked batch
            terms = csmae_loss(model(batch), batch, config.training.temperature)
            # Sum each individual loss term for batch averaging later
            for name, value in terms.items():
                totals[name] += float(value)
            batches += 1
    # Return averaged dictionary of all evaluation loss components
    return {name: value / max(batches, 1) for name, value in totals.items()}
