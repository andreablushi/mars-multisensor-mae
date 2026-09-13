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
        metrics: Every loss term averaged over the batches, keyed as the loss
            names them. The mask is drawn from the seed again on every call, so
            one epoch's loss is the next one's to beat.
    """
    model.eval()
    generator = torch.Generator(device=device).manual_seed(config.dataset.seed)
    totals = defaultdict(float)
    batches = 0
    with torch.no_grad():
        for batch, _ in loader:
            batch = {name: tokens.to(device) for name, tokens in batch.items()}
            batch = random_correspondence(batch, config.training.mask_ratio, generator)
            terms = csmae_loss(model(batch), batch, config.training.temperature)
            for name, value in terms.items():
                totals[name] += float(value)
            batches += 1
    return {name: value / max(batches, 1) for name, value in totals.items()}
