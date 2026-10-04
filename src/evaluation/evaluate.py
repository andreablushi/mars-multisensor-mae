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
    """One tile's tokens, where each falls and how much each weighs.

    Attributes:
        values: The token vectors, of unit length. (S, D)
        offset: The east, north and delay cell each token falls in, the delay from
            the tile's surface. (S, 3)
        weight: What each token weighs, the same for every instrument and zero for
            a token not compared. (S,)
    """

    values: Tensor
    offset: Tensor
    weight: Tensor


def evaluate_latent_space(
    model: CrossSensorMAE,
    loader: DataLoader,
    device: torch.device,
    cell_m: float,
    delay_rows: int,
    delay_window: tuple[int, int],
) -> dict[str, TileTokens]:
    """Return the tokens of every tile of one build, over a slab around its surface.

    Args:
        model: The model, loaded from a checkpoint and on the device.
        loader: The tiles to read, in batches, each tile read whole.
        device: Where the model runs.
        cell_m: How far a cell tokens are matched by runs along the ground, in metres.
        delay_rows: How many radar delay rows a cell spans.
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
            # A tile's surface cell is the median delay of its patches spanning none
            ground = counted & (placed[..., 5] == 0)  # (B, S)
            rows = placed[..., 2].masked_fill(~ground, torch.nan)  # (B, S)
            rows = rows.nanmedian(dim=1).values  # (B,)
            surface = (rows / delay_rows).floor().long()  # (B,)
            shift = torch.stack(
                [torch.zeros_like(surface)] * 2 + [surface], dim=-1
            )  # (B, 3)
            size = placed.new_tensor([cell_m, cell_m, delay_rows])
            offset = (placed[..., :3] / size).floor().long() - shift[:, None]
            first, last = delay_window
            slab = counted & (offset[..., 2] >= first) & (offset[..., 2] <= last)
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
                    values[at].float(), offset[at], weight[at].float()
                )
    return tiles
