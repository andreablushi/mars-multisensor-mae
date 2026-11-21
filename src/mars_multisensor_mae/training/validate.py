"""Measuring a model over one split."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from mars_multisensor_mae.architecture.mae import CrossSensorMAE
from mars_multisensor_mae.architecture.models.context import Pooling
from mars_multisensor_mae.training.step import batch_step


def validation_terms(
    model: CrossSensorMAE,
    loader: DataLoader,
    mask_ratio: float,
    aligned: Sequence[str],
    pooling: Pooling,
    temperature: float,
    uniformity_weight: float,
    generators: Mapping[str, torch.Generator],
    device: torch.device,
) -> dict[str, float]:
    """Return the loss terms over one split.

    Args:
        model: The model, on the device.
        loader: The split, in batches.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        aligned: The instruments MIM aligns, none for no MIM.
        pooling: What MIM aligns of each instrument, its GAP or its CLS.
        temperature: What MIM divides every cosine by.
        uniformity_weight: What the mean uniformity is multiplied by in the loss.
        generators: What fixes each instrument's masks, on the device.
        device: Where the model runs.

    Returns:
        metrics: Every loss term, averaged over the split's batches.
    """
    with torch.no_grad():
        terms = [
            batch_step(
                model,
                batch,
                mask_ratio,
                aligned,
                pooling,
                temperature,
                uniformity_weight,
                generators,
                device,
            )
            for batch, _ in loader
        ]
    return {
        name: float(torch.stack([one[name] for one in terms]).mean())
        for name in terms[0]
    }
