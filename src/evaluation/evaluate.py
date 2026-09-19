"""Measuring what a trained model's latent space made of the classes it never read."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from umap import UMAP

from architecture.mae import CrossSensorMAE
from evaluation.metrics import (
    class_similarity,
    grid_metrics,
    reconstruction_metrics,
    retrieval_metrics,
    silhouette_metrics,
    similarity_metrics,
    tile_similarity,
)
from training.masking import masked_reconstruction


@dataclass(frozen=True, slots=True)
class Evaluation:
    """What one pass over a split made of its latent space.

    Attributes:
        metrics: Every measured number, keyed as it is logged.
        intervals: Half the 95% interval around each mean, zero for a count.
        similarity: The mean matched similarity of every ordered pair of classes. (C, C)
        names: The classes, in the order the matrix holds them.
        placed: Where each measured tile sits on the plane. (N, 2)
        classes: The class of each of them, in that same order.
    """

    metrics: dict[str, float]
    intervals: dict[str, float]
    similarity: np.ndarray
    names: list[str]
    placed: np.ndarray
    classes: list[str]


def evaluate_latent_space(
    model: CrossSensorMAE,
    loader: DataLoader,
    neighbours: int,
    seed: int,
    device: torch.device,
) -> Evaluation:
    """Return what the model's latent space makes of one split's classes.

    Args:
        model: The model, loaded from a checkpoint and on the device.
        loader: The split to read, in batches, each tile read whole.
        neighbours: How many nearest tiles one retrieval reads.
        seed: What fixes the layout on the plane.
        device: Where the model runs.

    Returns:
        evaluation: Every metric, the class similarities, and the plane.
    """
    model.eval()
    embedded, read = [], []
    with torch.no_grad():
        for batch, cells, identities in loader:
            batch = {name: tokens.to(device) for name, tokens in batch.items()}
            grid = model.embed(batch, cells.to(device))
            for at, (one, _) in enumerate(identities):
                embedded.append(grid.values[at][grid.occupied[at]])  # (Q, D)
                read.append(one)
    # A tile holding no patch of any instrument the model reads reaches no cell.
    kept = [(one, held) for one, held in zip(read, embedded, strict=True) if len(held)]
    classes = [one for one, _ in kept]
    cells = [held for _, held in kept]
    matched = tile_similarity(cells)  # (N, N)
    distance = (1 - matched).clamp(min=0).numpy()  # (N, N)
    np.fill_diagonal(distance, 0)
    similarity, names = class_similarity(matched, classes)  # (C, C)
    # Evaluate retrieval, silhouette and similarity of the representations
    measured = (
        retrieval_metrics(matched, classes, neighbours)
        | silhouette_metrics(distance, classes)
        | similarity_metrics(similarity)
        | grid_metrics(cells)
    )
    # Package the metrics, their confidence bounds, and the 2D UMAP layout
    return Evaluation(
        metrics={name: value for name, (value, _) in measured.items()}
        | {
            "tiles": len(classes),
            "classes": len(names),
            "empty": len(read) - len(classes),
        },
        intervals={name: half for name, (_, half) in measured.items()},
        similarity=similarity,
        names=names,
        # Map the tiles into 2D space from the distances already measured
        placed=UMAP(
            n_components=2, metric="precomputed", random_state=seed
        ).fit_transform(distance),  # (N, 2)
        classes=classes,
    )


def evaluate_reconstruction(
    model: CrossSensorMAE,
    loader: DataLoader,
    mask_ratio: float,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    """Return how well the model rebuilds the patches it was never shown.

    Args:
        model: The model, loaded from a checkpoint and on the device.
        loader: The split to read, in batches.
        mask_ratio: How much of each instrument is hidden.
        seed: What fixes the masks.
        device: Where the model runs.

    Returns:
        metrics: Every "umr/<sensor>/<name>" and "cmr/<sensor>/<name>", batch averaged.
    """
    model.eval()  # Switch model to evaluation mode
    # Seed generator for reproducible masking
    generator = torch.Generator(device=device).manual_seed(seed)
    # Accumulate metric totals across batches
    totals: defaultdict[str, float] = defaultdict(float)
    batches = 0
    # Disable autograd to reduce memory usage and speed up execution
    with torch.no_grad():
        for batch, cells, _ in loader:
            # Generate patch reconstructions across all sensor pairings
            batch, reconstruction = masked_reconstruction(
                model, batch, cells, mask_ratio, generator, device
            )
            # Count available cross-modal sources for averaging
            others = max(len(batch) - 1, 1)
            for asked, tokens in batch.items():
                # Identify valid patches that were masked out
                hidden = tokens.present & ~tokens.visible  # (B, K)
                for read in batch:
                    # Check the grid the decoder reads holds a cell, as the loss does
                    readable = reconstruction.grids[read].occupied.any(
                        dim=1, keepdim=True
                    )  # (B, 1)
                    own = read == asked
                    # Evaluate reconstruction accuracy on target hidden patches
                    measured = reconstruction_metrics(
                        reconstruction.predictions[asked, read],
                        tokens.values,
                        tokens.valid,
                        hidden if own else hidden & readable,
                    )
                    # Accumulate by unimodal (umr) vs cross-modal (cmr)
                    for name, value in measured.items():
                        key = f"{'umr' if own else 'cmr'}/{asked}/{name}"
                        totals[key] += value if own else value / others
            batches += 1
    # Return averaged metric scores across all batches
    return {name: value / max(batches, 1) for name, value in totals.items()}
