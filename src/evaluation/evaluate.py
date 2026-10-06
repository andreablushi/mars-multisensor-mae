"""Embedding the tiles a trained model never read."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from architecture.mae import CrossSensorMAE
from training.step import device_batch

# B = batch, S = tokens, D = token channels.


@dataclass(frozen=True, slots=True)
class TileTokens:
    """One tile's tokens, where each sits and how much each weighs.

    Attributes:
        values: The token vectors, of unit length. (S, D)
        ground: How far east and north of the tile centre each sits, in metres. (S, 2)
        weight: What each token weighs, the same for every instrument and zero for
            a token not compared. (S,)
    """

    values: Tensor
    ground: Tensor
    weight: Tensor


def evaluate_latent_space(
    model: CrossSensorMAE,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, TileTokens]:
    """Return the tokens of every tile of one build, each instrument weighing the same.

    Args:
        model: The model, loaded from a checkpoint and on the device.
        loader: The tiles to read, in batches, each tile read whole.
        device: Where the model runs.

    Returns:
        tiles: One set of tokens per tile, keyed by the tile it stands for.
    """
    model.eval()
    tiles = {}
    with torch.no_grad():
        for batch, identities in loader:
            batch = device_batch(batch, device)
            present = {name: tokens.present for name, tokens in batch.items()}  # (B, K)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                values, placed, counted = model.embed(batch, present)
            # Each instrument weighs the same, however many tokens it holds
            parts = counted.split([one.position.shape[1] for one in batch.values()], 1)
            counts = torch.stack([part.sum(dim=1) for part in parts], dim=1)  # (B, I)
            instruments = (counts > 0).sum(dim=1, keepdim=True)  # (B, 1)
            share = 1 / (counts * instruments).clamp(min=1)  # (B, I)
            weight = torch.cat(
                [part * share[:, [at]] for at, part in enumerate(parts)], dim=1
            )  # (B, S)
            for at, identity in enumerate(identities):
                # Kept in full precision, since the distances are read in its dtype
                tiles[identity] = TileTokens(
                    values[at].float(), placed[at, :, :2], weight[at].float()
                )
    return tiles
