"""Embedding the tiles a trained model never read."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader

from architecture.mae import CrossSensorMAE
from training.step import device_batch


def evaluate_latent_space(
    model: CrossSensorMAE, loader: DataLoader, device: torch.device
) -> dict[str, dict[str, np.ndarray]]:
    """Return one vector per tile and instrument: its tokens averaged, of unit length.

    Args:
        model: The model, on the device.
        loader: The tiles to read, in batches, each tile read whole.
        device: Where the model runs.

    Returns:
        vectors: Each instrument's vector of every tile holding it, keyed by
            instrument, then by tile. (D,)
    """
    model.eval()
    vectors = defaultdict(dict)
    with torch.no_grad():
        for batch, identities in loader:
            batch = device_batch(batch, device)
            present = {name: tokens.present for name, tokens in batch.items()}  # (B, K)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                embedded = model.embed(batch, present)
            for name, tokens in embedded.items():
                held = present[name].unsqueeze(-1)  # (B, K, 1)
                summed = (tokens.float() * held).sum(dim=1)  # (B, D)
                pooled = functional.normalize(summed, dim=-1).cpu().numpy()
                for at, identity in enumerate(identities):
                    if present[name][at].any():
                        vectors[name][identity] = pooled[at]
    return dict(vectors)
