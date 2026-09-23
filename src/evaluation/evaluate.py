"""Embedding the tiles a trained model never read."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from architecture.mae import CrossSensorMAE
from architecture.models import TileGrid


def evaluate_latent_space(
    model: CrossSensorMAE, loader: DataLoader, device: torch.device
) -> dict[str, TileGrid]:
    """Return the grid standing for every tile of one build, none of it hidden.

    Args:
        model: The model, loaded from a checkpoint and on the device.
        loader: The tiles to read, in batches, each tile read whole.
        device: Where the model runs.

    Returns:
        grids: One grid per tile, keyed by the tile it stands for.
    """
    model.eval()
    grids = {}
    with torch.no_grad():
        for batch, cells, identities in loader:
            batch = {name: tokens.to(device) for name, tokens in batch.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16):
                grid = model.embed(batch, cells.to(device))
            for at, identity in enumerate(identities):
                # Kept in full precision, since the distances are read in its dtype
                grids[identity] = TileGrid(
                    grid.values[at].float(), grid.occupied[at], grid.offset[at]
                )
    return grids
