"""One batch on the model device: every protocol's prediction, and the points scored."""

from __future__ import annotations

import math
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


def scored_swim_points(
    batch: dict[str, Tokens], points: Tensor, reach_m: float
) -> Tensor:
    """Return which of SWIM's points each tile of a batch is scored on.

    Args:
        batch: Each instrument's patches over the batch, as the loader hands it.
        points: Where SWIM's points sit on a tile, east and north in metres. (P * P, 2)
        reach_m: How near a read patch a point is scored, in metres.

    Returns:
        scored: Whether each point lies within reach of a patch the tile was cut to.
            (B, P * P)
    """
    # A point is scored where a patch the tile was cut to stands near it
    spots = torch.cat([one.position[..., :2] for one in batch.values()], 1)
    read = torch.cat([one.present for one in batch.values()], 1)  # (B, S)
    apart = torch.cdist(points.expand(len(spots), -1, -1), spots)  # (B, P * P, S)
    return apart.masked_fill(~read[:, None], math.inf).amin(-1) <= reach_m
