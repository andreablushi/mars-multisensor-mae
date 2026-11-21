"""Predicting every protocol's points over one split, as the test reads it."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from benchmarking.architecture.model import PROTOCOLS, BenchmarkModel
from benchmarking.evaluation.predict import crop_mean_predictions


def validate(
    models: Mapping[str, BenchmarkModel],
    loader: DataLoader,
    device: torch.device,
) -> dict[str, dict[str, np.ndarray]]:
    """Return each model's prediction of every point over one split.

    Args:
        models: Each protocol's model, on the device.
        loader: The split, in batches, each tile read as the budget crops covering
            SWIM's points.
        device: Where the models run.

    Returns:
        predictions: Every point's prediction of each target, the mean of the crops
            reaching it, keyed by protocol, then by tile. (P * P, Z)
    """
    still = next(protocol for protocol, tuned in PROTOCOLS.items() if not tuned)
    predictions = {protocol: {} for protocol in models}
    for model in models.values():
        model.eval()
    with torch.no_grad():
        for batch, identities, scored in loader:
            held = crop_mean_predictions(
                {"validation": models}, batch, identities, scored, still, device
            )["validation"]
            for protocol, by_tile in held.items():
                predictions[protocol].update(by_tile)
    return predictions
