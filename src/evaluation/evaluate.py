"""Embedding the tiles a trained model never read."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader

from architecture.mae import CrossSensorMAE
from architecture.tokens import CLS, GAP, Tokens, global_average_pooling

EVERY_INSTRUMENT = "all"


def evaluate(
    model: CrossSensorMAE, loader: DataLoader, device: torch.device
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Return one vector per tile, pooling and view: each instrument's, and all.

    Args:
        model: The model, on the device.
        loader: The tiles to read, in batches, each tile read whole.
        device: Where the model runs.

    Returns:
        vectors: Each tile's unit vector, keyed by CLS or GAP, then by instrument or
            EVERY_INSTRUMENT, then by tile, CLS of every instrument being the
            model's own and GAP's the mean of the instruments'. (D,)
    """
    model.eval()
    vectors = defaultdict(lambda: defaultdict(dict))
    with torch.no_grad():
        for batch, identities in loader:
            batch = {
                name: Tokens(*(one.to(device, non_blocking=True) for one in tokens))
                for name, tokens in batch.items()
            }
            present = {name: tokens.present for name, tokens in batch.items()}
            cls, tokens = model.shared_tokens(batch, present)  # (B, 1 + K, D) each
            pooled = {
                CLS: {
                    name: functional.normalize(held[:, 0], dim=-1)
                    for name, held in tokens.items()
                },
                GAP: {
                    name: global_average_pooling(held[:, 1:], present[name])
                    for name, held in tokens.items()
                },
            }
            pooled[CLS][EVERY_INSTRUMENT] = functional.normalize(cls, dim=-1)
            stacked = torch.stack(list(pooled[GAP].values()), dim=1)  # (B, M, D)
            pooled[GAP][EVERY_INSTRUMENT] = global_average_pooling(
                stacked, torch.ones(stacked.shape[:2], dtype=torch.bool, device=device)
            )
            for pooling, by_view in pooled.items():
                for view, held in by_view.items():
                    vectors[pooling][view].update(
                        zip(identities, held.cpu().numpy(), strict=True)
                    )
    return {pooling: dict(by_view) for pooling, by_view in vectors.items()}
