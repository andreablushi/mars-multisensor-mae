"""Measuring a model over one split."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from architecture.mae import CrossSensorMAE


def validation_terms(
    model: CrossSensorMAE,
    loader: DataLoader,
    step: Callable[..., dict[str, Tensor]],
    generators: Mapping[str, torch.Generator],
) -> dict[str, float]:
    """Return the loss terms over one split.

    Args:
        model: The model.
        loader: The split, in batches.
        step: One batch's loss terms, from the batch and the mask generators.
        generators: What fixes each instrument's masks, on the device.

    Returns:
        metrics: Every loss term, averaged over the split's batches.
    """
    with torch.no_grad():
        terms = [step(batch, generators=generators) for batch, _ in loader]
    return {
        name: float(torch.stack([one[name] for one in terms]).mean())
        for name in terms[0]
    }
