"""Embedding the tiles a trained model never read."""

from __future__ import annotations

from collections import defaultdict
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
    sizes: Mapping[str, Mapping[str, int]],
    delay_window: tuple[int, int],
    device: torch.device,
) -> dict[str, dict[str, TileTokens]]:
    """Return the tokens of every tile of one build, each instrument kept apart.

    Args:
        model: The model, loaded from a checkpoint and on the device.
        loader: The tiles to read, in batches, each tile read whole.
        sizes: How far a patch of each instrument runs along each axis it is cut on.
        delay_window: The first and last delay cell a sounder token is kept in,
            counted from the cell of the surface beneath it.
        device: Where the model runs.

    Returns:
        tiles: Each instrument's tokens of every tile, keyed by instrument, then by
            the tile they stand for.

    Raises:
        ValueError: When a tile holds no ground token, or no kept token of an
            instrument the model reads.
    """
    first, last = delay_window
    model.eval()
    tiles = defaultdict(dict)
    with torch.no_grad():
        for batch, identities in loader:
            batch = device_batch(batch, device)
            present = {name: tokens.present for name, tokens in batch.items()}  # (B, K)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                embedded = model.embed(batch, present)
            for at, identity in enumerate(identities):
                # A ground token sits at the row its surface echo lands on
                ground = torch.cat(
                    [
                        batch[name].position[at, present[name][at]]
                        for name in batch
                        if Axis.DELAY not in sizes[name]
                    ]
                )  # (G, 3)
                if not len(ground):
                    raise ValueError(f"{identity} holds no ground token.")
                for name, values in embedded.items():
                    held = present[name][at]  # (K,)
                    position = batch[name].position[at]  # (K, 3)
                    if Axis.DELAY in sizes[name]:
                        # Kept near the surface beneath its nearest ground token
                        nearest = torch.cdist(position[:, :2], ground[:, :2]).argmin(1)
                        rows = sizes[name][Axis.DELAY]
                        depth = (position[:, 2] / rows).floor() - (
                            ground[nearest, 2] / rows
                        ).floor()  # (K,)
                        held = held & (depth >= first) & (depth <= last)
                    if not held.any():
                        raise ValueError(f"{identity} holds no kept {name} token.")
                    # Kept in full precision, since the distances are read in its dtype
                    tiles[name][identity] = TileTokens(
                        values[at, held].float(), position[held, :2]
                    )
    return dict(tiles)
