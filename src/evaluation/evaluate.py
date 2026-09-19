"""Measuring what a trained model's latent space made of the tiles it never read."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from architecture.mae import CrossSensorMAE
from architecture.models import TileGrid


def embed_split(
    model: CrossSensorMAE, loader: DataLoader, device: torch.device
) -> dict[str, TileGrid]:
    """Return the grid standing for every tile of one split, none of it hidden.

    Args:
        model: The model, loaded from a checkpoint and on the device.
        loader: The split to read, in batches, each tile read whole.
        device: Where the model runs.

    Returns:
        grids: One grid per tile, keyed by the tile it stands for.
    """
    model.eval()
    grids = {}
    with torch.no_grad():
        for batch, cells, identities in loader:
            batch = {name: tokens.to(device) for name, tokens in batch.items()}
            grid = model.embed(batch, cells.to(device))
            for at, identity in enumerate(identities):
                grids[identity] = TileGrid(
                    grid.values[at], grid.occupied[at], grid.offset[at]
                )
    return grids


def evaluate_latent_space(
    model: CrossSensorMAE, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    """Return what the model's latent space makes of one split.

    Nothing is measured yet. What a tile should be matched against is a
    similarity between geological features, and that dataset is not built.

    Args:
        model: The model, loaded from a checkpoint and on the device.
        loader: The split to read, in batches, each tile read whole.
        device: Where the model runs.

    Returns:
        metrics: How many tiles were embedded, and nothing measured over them.
    """
    grids = embed_split(model, loader, device)
    return {"tiles": float(len(grids))}
