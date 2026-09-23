"""Measuring what a trained model's latent space made of the tiles it never read."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from architecture.mae import CrossSensorMAE
from architecture.models import TileGrid
from evaluation.metrics import (
    chamfer_distances,
    class_distances,
    class_separation,
    retrieval_metrics,
    silhouette_by_class,
    umap_projection,
)


@dataclass(frozen=True, slots=True)
class LatentMeasure:
    """What one model's latent space came to over the labelled tiles.

    Attributes:
        metrics: Every number measured, keyed as it is logged.
        classes: The classes, in the order the distances hold them.
        distances: The mean distance between the tiles of two classes. (C, C)
        tiles: The tiles, in the order the tile distances hold them.
        tile_distances: The distance between every pair of tiles. (T, T)
        projection: Where UMAP lays each tile on a plane, in the same order. (T, 2)
    """

    metrics: dict[str, float]
    classes: list[str]
    distances: np.ndarray
    tiles: list[str]
    tile_distances: np.ndarray
    projection: np.ndarray


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


def measure_latent_space(
    grids: Mapping[str, TileGrid],
    classes: Mapping[str, str],
    minimal_chamfer_cell_distance: int | None,
    seed: int,
) -> LatentMeasure:
    """Return what the grids came to, read against the class each tile carries.

    Args:
        grids: One grid per tile, keyed by the tile it stands for.
        classes: The class each of those tiles earned, keyed the same way.
        minimal_chamfer_cell_distance: How far, in cells, a cell may be matched
            from its own offset.
        seed: What the UMAP layout is drawn with.

    Returns:
        measured: Every number the latent space came to, the class and tile distances.
    """
    tiles = sorted(grids)
    labels = [classes[tile] for tile in tiles]
    held = chamfer_distances(
        [grids[tile] for tile in tiles], minimal_chamfer_cell_distance
    )
    distances = held.double().cpu().numpy()  # (T, T)
    order, matrix = class_distances(distances, labels)
    return LatentMeasure(
        metrics=retrieval_metrics(distances, labels)
        | silhouette_by_class(distances, labels)
        | class_separation(matrix),
        classes=order,
        distances=matrix,
        tiles=tiles,
        tile_distances=distances,
        projection=umap_projection(distances, seed),
    )
