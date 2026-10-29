"""Measuring a model over one split."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import DataLoader

from architecture.mae import CrossSensorMAE
from training.step import mask_generators, step_terms


def validation_terms(
    model: CrossSensorMAE,
    loader: DataLoader,
    mask_ratio: float,
    aligned: Sequence[str],
    pooling: str,
    temperature: float,
    measure: str,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    """Return the loss terms over one split.

    Args:
        model: The model, which is switched to evaluation.
        loader: The split, in batches.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        aligned: The instruments MIM aligns, none for no MIM.
        pooling: What MIM aligns of each instrument, its GAP or its CLS.
        temperature: What MIM divides every cosine by.
        measure: The reconstruction error, MSE or RMSE.
        seed: What fixes the masks.
        device: Where the model runs.

    Returns:
        metrics: Every loss term, averaged over the split's batches.
    """
    model.eval()
    generators = mask_generators(model.encoders, seed, device)
    with torch.no_grad():
        terms = [
            step_terms(
                model,
                batch,
                mask_ratio,
                aligned,
                pooling,
                temperature,
                measure,
                generators,
                device,
            )
            for batch, _ in loader
        ]
    return {
        name: float(torch.stack([one[name] for one in terms]).mean())
        for name in terms[0]
    }
