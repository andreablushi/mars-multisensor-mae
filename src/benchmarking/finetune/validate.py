"""Predicting a head's chance of ice over one split."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from architecture.mae import CrossSensorMAE
from benchmarking.finetune.step import batch_logits


def predicted_probabilities(
    model: CrossSensorMAE, head: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, np.ndarray]:
    """Return a head's chance of ice over one split, the model and head left training.

    Args:
        model: The model, measured in evaluation on the device.
        head: The classifier or the segmenter, measured alike.
        loader: The split, in batches, each tile read whole.
        device: Where the model runs.

    Returns:
        probabilities: The chance of ice at each depth, keyed by the tile.
            (Z,) or (P * P, Z)
    """
    model.eval()
    head.eval()
    probabilities = {}
    with torch.no_grad():
        for batch, identities in loader:
            chances = batch_logits(model, head, batch, device).sigmoid()
            probabilities.update(zip(identities, chances.cpu().numpy(), strict=True))
    model.train()
    head.train()
    return probabilities
