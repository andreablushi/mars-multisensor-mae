"""Embedding the tiles a trained model never read."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from architecture.grid import TileGrid
from architecture.mae import CrossSensorMAE


def evaluate_latent_space(
    model: CrossSensorMAE,
    loader: DataLoader,
    device: torch.device,
    delay_rows: int,
    delay_window: tuple[int, int],
) -> dict[str, TileGrid]:
    """Return the grid of every tile of one build, over a slab around its surface.

    Args:
        model: The model, loaded from a checkpoint and on the device.
        loader: The tiles to read, in batches, each tile read whole.
        device: Where the model runs.
        delay_rows: How many radar delay rows a cell spans.
        delay_window: The first and last delay cell kept, counted from the surface.

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
            offset = grid.offset - shift[:, None]  # (B, Q, 3)
            first, last = delay_window
            slab = (offset[..., 2] >= first) & (offset[..., 2] <= last)  # (B, Q)
            for at, identity in enumerate(identities):
                # Kept in full precision, since the distances are read in its dtype
                grids[identity] = TileGrid(
                    grid.values[at].float(),
                    grid.occupied[at] & slab[at],
                    offset[at],
                    grid.position[at],
                )
    return grids
