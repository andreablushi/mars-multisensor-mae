"""Embedding the tiles a trained model never read."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from architecture.mae import CrossSensorMAE
from training.step import device_batch

# B = batch, S = tokens, D = token channels.


@dataclass(frozen=True, slots=True)
class TileTokens:
    """One instrument's tokens of one tile, and where each sits.

    Attributes:
        values: The token vectors, of unit length. (S, D)
        ground: How far east and north of the tile centre each sits, in metres. (S, 2)
    """

    values: Tensor
    ground: Tensor


def evaluate_latent_space(
    model: CrossSensorMAE,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, dict[str, TileTokens]]:
    """Return the tokens of every tile of one build, each instrument kept apart.

    Args:
        model: The model, loaded from a checkpoint and on the device.
        loader: The tiles to read, in batches, each tile read whole.
        device: Where the model runs.

    Returns:
        tiles: Each instrument's tokens of every tile, keyed by instrument, then by
            the tile they stand for.

    Raises:
        ValueError: When a tile holds no token of an instrument the model reads.
    """
    model.eval()
    tiles = defaultdict(dict)
    with torch.no_grad():
        for batch, identities in loader:
            batch = device_batch(batch, device)
            present = {name: tokens.present for name, tokens in batch.items()}  # (B, K)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                embedded = model.embed(batch, present)
            for name, values in embedded.items():
                for at, identity in enumerate(identities):
                    held = present[name][at]  # (K,)
                    if not held.any():
                        raise ValueError(f"{identity} holds no {name} token.")
                    # Kept in full precision, since the distances are read in its dtype
                    tiles[name][identity] = TileTokens(
                        values[at, held].float(), batch[name].position[at, held, :2]
                    )
    return dict(tiles)
