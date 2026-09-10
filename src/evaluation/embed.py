"""Embedding features, one unit vector each."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from architecture.mae import CrossSensorMAE
from architecture.swath import feature_swath
from architecture.tokens import Tokens


def embed_batch(model: CrossSensorMAE, batch: dict[str, Tokens]) -> Tensor:
    """Return one unit vector per feature of a batch.

    Args:
        model: The model, in evaluation.
        batch: Each instrument's patches over the batch, on the model's device.

    Returns:
        embeddings: The mean direction of every present token of every
            instrument of each feature. (B, L)
    """
    tokens = model.encode(batch)
    directions = torch.cat([tokens[name] for name in batch], dim=1)  # (B, sum K, L)
    present = torch.cat([batch[name].present for name in batch], dim=1)  # (B, sum K)
    return feature_swath(directions, present)  # (B, L)


def embed_features(
    model: CrossSensorMAE, loader: DataLoader, device: torch.device
) -> tuple[Tensor, list[str]]:
    """Return one unit vector per feature of a split, and each feature's class.

    Args:
        model: The model, which is switched to evaluation.
        loader: The split, in batches.
        device: Where the model runs.

    Returns:
        embeddings: One per feature, in the loader's order. (N, L)
        classes: The class of each, in the same order.
    """
    model.eval()
    embeddings, classes = [], []
    with torch.no_grad():
        for batch, labels in loader:
            batch = {name: tokens.to(device) for name, tokens in batch.items()}
            embeddings.append(embed_batch(model, batch).cpu())
            classes.extend(labels)
    return torch.cat(embeddings), classes
