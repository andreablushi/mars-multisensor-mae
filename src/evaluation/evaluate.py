"""Embedding the tiles a trained model never read."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from building.common.layout import Axis
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
    axes: Mapping[str, tuple[str, ...]],
    device: torch.device,
    delay_rows: int,
    delay_window: tuple[int, int],
) -> dict[str, TileTokens]:
    """Return the tokens of every tile of one build, over a slab around its surface.

    Args:
        model: The model, loaded from a checkpoint and on the device.
        loader: The tiles to read, in batches, each tile read whole.
        axes: What each axis of each instrument's values holds.
        device: Where the model runs.
        delay_rows: How many radar delay rows a delay cell spans.
        delay_window: The first and last delay cell kept, counted from the surface.

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
            # A tile's surface cell is the median delay of its ground patches
            surface = torch.cat(
                [
                    one.present & (Axis.DELAY not in axes[name])
                    for name, one in batch.items()
                ],
                dim=1,
            )  # (B, S)
            rows = placed[..., 2].masked_fill(~surface, torch.nan)  # (B, S)
            rows = rows.nanmedian(dim=1, keepdim=True).values  # (B, 1)
            # The delay cell of each token, counted from its tile's surface cell
            depth = (placed[..., 2] / delay_rows).floor() - (rows / delay_rows).floor()
            first, last = delay_window
            slab = counted & (depth >= first) & (depth <= last)  # (B, S)
            # Each instrument weighs the same, however many tokens it holds
            parts = slab.split([one.position.shape[1] for one in batch.values()], 1)
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
