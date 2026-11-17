"""One batch of test crops on the model device: each tile's prediction per model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from architecture.models.tokens import Tokens
from benchmarking.architecture.model import PROTOCOLS, BenchmarkModel
from training.step import autocast_dtype


def crop_mean_predictions(
    models: Mapping[str, Mapping[str, BenchmarkModel]],
    batch: dict[str, Tokens],
    identities: Sequence[str],
    scored: np.ndarray,
    still: str,
    device: torch.device,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Return each tile's prediction by every model, a point the mean of its crops'.

    Args:
        models: Each share's model of each protocol, on the device.
        batch: Each instrument's patches over every crop, a tile's crops side by side.
        identities: The tile each crop belongs to.
        scored: Which points each crop predicts. (C, P * P)
        still: The protocol whose MAE is frozen, read once for every share.
        device: Where the models run.

    Returns:
        predictions: Each tile's prediction of every point and target, keyed by
            share, protocol and tile. (P * P, Z)
    """
    # Each point is the mean of the crops of its tile that predict it
    weights = torch.as_tensor(scored, dtype=torch.float32).unsqueeze(-1)
    crops = {
        tile: [at for at, one in enumerate(identities) if one == tile]
        for tile in dict.fromkeys(identities)
    }
    batch = {name: tokens.to(device) for name, tokens in batch.items()}
    predictions = {share: {} for share in models}
    with torch.autocast(device.type, dtype=autocast_dtype(device)):
        # The frozen MAE is the same under every share, so it reads once
        frozen = next(iter(models.values()))[still].tokens(batch)
        for share, held in models.items():
            for protocol, model in held.items():
                read_tokens = model.tokens(batch) if PROTOCOLS[protocol] else frozen
                predicted = model.probe(read_tokens).float().cpu() * weights
                predictions[share][protocol] = {
                    tile: (predicted[at].sum(0) / weights[at].sum(0)).numpy()
                    for tile, at in crops.items()
                }
    return predictions
