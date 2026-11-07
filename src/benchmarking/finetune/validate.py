"""Predicting every head's consistency over one split."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from architecture.mae import CrossSensorMAE
from benchmarking.finetune.step import batch_consistency


def predicted_consistency(
    model: CrossSensorMAE,
    heads: Mapping[str, nn.Module],
    loader: DataLoader,
    device: torch.device,
) -> dict[str, dict[str, np.ndarray]]:
    """Return every head's consistency over one split, all of them left training.

    Args:
        model: The model, measured in evaluation on the device.
        heads: The classifier and the segmenter, measured alike, keyed by task.
        loader: The split, in batches, each tile read whole.
        device: Where the model runs.

    Returns:
        consistency: The consistency at each depth, keyed by task, then by tile.
            (Z,) or (P * P, Z)
    """
    parts = [model, *heads.values()]
    for part in parts:
        part.eval()
    consistency = {task: {} for task in heads}
    with torch.no_grad():
        for batch, identities in loader:
            for task, held in batch_consistency(model, heads, batch, device)[0].items():
                consistency[task].update(
                    zip(identities, held.cpu().numpy(), strict=True)
                )
    for part in parts:
        part.train()
    return consistency
