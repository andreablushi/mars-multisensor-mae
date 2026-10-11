"""Embedding the tiles a trained model never read."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from architecture.mae import CrossSensorMAE
from architecture.tokens import Tokens
from evaluation.similarity import global_average_pooling

EVERY_INSTRUMENT = "all"


def evaluate(
    model: CrossSensorMAE, loader: DataLoader, device: torch.device
) -> dict[str, dict[str, np.ndarray]]:
    """Return one vector per tile and view: each instrument's, and all of them.

    Args:
        model: The model, on the device.
        loader: The tiles to read, in batches, each tile read whole.
        device: Where the model runs.

    Returns:
        vectors: Each tile's unit vector, keyed by instrument or EVERY_INSTRUMENT,
            then by tile. (D,)
    """
    model.eval()
    vectors = defaultdict(dict)
    with torch.no_grad():
        for batch, identities in loader:
            batch = {
                name: Tokens(*(one.to(device, non_blocking=True) for one in tokens))
                for name, tokens in batch.items()
            }
            present = {name: tokens.present for name, tokens in batch.items()}
            pooled = {
                name: global_average_pooling(tokens, present[name])
                for name, tokens in model.shared_tokens(batch, present).items()
            }
            stacked = torch.stack(list(pooled.values()), dim=1)  # (B, M, D)
            pooled[EVERY_INSTRUMENT] = global_average_pooling(
                stacked, torch.ones(stacked.shape[:2], dtype=torch.bool, device=device)
            )
            for view, held in pooled.items():
                vectors[view].update(zip(identities, held.cpu().numpy(), strict=True))
    return dict(vectors)
