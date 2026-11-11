"""Predicting every protocol's points over one split."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from benchmarking.architecture.model import BenchmarkModel
from benchmarking.finetune.step import batch_step


def validate(
    models: Mapping[str, BenchmarkModel],
    loader: DataLoader,
    device: torch.device,
) -> dict[str, dict[str, np.ndarray]]:
    """Return each model's prediction of every point over one split.

    Args:
        models: Each protocol's model, on the device.
        loader: The split, in batches, each tile read whole.
        device: Where the models run.

    Returns:
        predictions: Every point's prediction of each target, keyed by protocol,
            then by tile. (P * P, Z)
    """
    predictions = {protocol: {} for protocol in models}
    with torch.no_grad():
        for batch, identities in loader:
            for protocol, held in batch_step(models, batch, device).items():
                predictions[protocol].update(
                    zip(identities, held.cpu().numpy(), strict=True)
                )
    return predictions
