"""One batch on the model device: every protocol's prediction."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from architecture.models.tokens import Tokens
from benchmarking.architecture.model import BenchmarkModel
from training.step import autocast_dtype


def batch_step(
    models: Mapping[str, BenchmarkModel],
    batch: dict[str, Tokens],
    device: torch.device,
) -> dict[str, Tensor]:
    """Return each model's prediction over one batch.

    Args:
        models: Each protocol's model, on the device.
        batch: Each instrument's patches over the batch, as the loader hands it.
        device: Where the models run.

    Returns:
        predictions: Every point's prediction of each target, in float, keyed by
            protocol. (B, P * P, Z)
    """
    batch = {name: tokens.to(device) for name, tokens in batch.items()}
    with torch.autocast(device.type, dtype=autocast_dtype(device)):
        return {protocol: model(batch).float() for protocol, model in models.items()}
