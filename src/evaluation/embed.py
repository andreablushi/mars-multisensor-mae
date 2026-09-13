"""Reading one split through a trained model, a latent vector per feature."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from architecture.mae import CrossSensorMAE


def embed_split(
    model: CrossSensorMAE, loader: DataLoader, device: torch.device
) -> tuple[Tensor, list[str]]:
    """Return the latent vector of every feature of one split, and its class.

    Args:
        model: The model, which is switched to evaluation.
        loader: The split, in batches.
        device: Where the model runs.

    Returns:
        latents: One vector per feature, over every patch it holds of every
            instrument, none hidden. (N, D)
        classes: The class of each of them, in that same order.
    """
    model.eval()
    held: list[Tensor] = []
    classes: list[str] = []
    with torch.no_grad():
        for batch, feature_classes in loader:
            batch = {name: tokens.to(device) for name, tokens in batch.items()}
            held.append(model.embed(batch).cpu())  # (B, D)
            classes.extend(feature_classes)
    return torch.cat(held), classes  # (N, D)
