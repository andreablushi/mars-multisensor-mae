"""Predicting the head's consistency over one split."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from architecture.mae import CrossSensorMAE
from benchmarking.finetune.step import batch_consistency


def predicted_consistency(
    model: CrossSensorMAE,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Return the head's consistency over one split, both left training.

    Args:
        model: The model, measured in evaluation on the device.
        head: The map head, measured alike.
        loader: The split, in batches, each tile read whole.
        device: Where the model runs.

    Returns:
        consistency: The consistency of every point at each depth, keyed by the
            tile. (P * P, Z)
    """
    model.eval()
    head.eval()
    consistency = {}
    with torch.no_grad():
        for batch, identities in loader:
            held = batch_consistency(model, head, batch, device)[0].cpu().numpy()
            consistency.update(zip(identities, held, strict=True))
    model.train()
    head.train()
    return consistency
