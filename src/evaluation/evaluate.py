"""Embedding the tiles a trained model never read."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from architecture.grid import TileGrid
from architecture.mae import CrossSensorMAE


def evaluate_latent_space(
    model: CrossSensorMAE, loader: DataLoader, device: torch.device, delay_rows: int
) -> dict[str, TileGrid]:
    """Return the grid standing for every tile of one build, its delay from its surface.

    Args:
        model: The model, loaded from a checkpoint and on the device.
        loader: The tiles to read, in batches, each tile read whole.
        device: Where the model runs.
        delay_rows: How many radar delay rows a cell spans.

    Returns:
        grids: One grid per tile, keyed by the tile it stands for.

    Raises:
        ValueError: When a tile holds no surface patch to count its delay from.
    """
    model.eval()
    grids = {}
    with torch.no_grad():
        for batch, cells, identities in loader:
            batch = {name: tokens.to(device) for name, tokens in batch.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16):
                grid = model.embed(batch, cells.to(device))
            # A tile's surface cell is the median delay of its patches spanning none
            held = list(batch.values())
            placed = torch.cat([one.position for one in held], dim=1)  # (B, K, 6)
            present = torch.cat([one.present for one in held], dim=1)  # (B, K)
            ground = present & (placed[..., 5] == 0)  # (B, K)
            rows = placed[..., 2].masked_fill(~ground, torch.nan)  # (B, K)
            rows = rows.nanmedian(dim=1).values  # (B,)
            if rows.isnan().any():
                raise ValueError("a tile holding no surface patch has no surface")
            surface = (rows / delay_rows).floor().long()  # (B,)
            shift = torch.stack(
                [torch.zeros_like(surface)] * 2 + [surface], dim=-1
            )  # (B, 3)
            for at, identity in enumerate(identities):
                # Kept in full precision, since the distances are read in its dtype
                grids[identity] = TileGrid(
                    grid.values[at].float(),
                    grid.occupied[at],
                    grid.offset[at] - shift[at],
                    grid.position[at],
                )
    return grids
