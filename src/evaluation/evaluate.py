"""Measuring what a trained model's latent space made of the classes it never read."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional
from torch.utils.data import DataLoader
from umap import UMAP

from architecture.mae import CrossSensorMAE
from architecture.models import FeatureGrid
from config.schema import Config
from evaluation.metrics import (
    class_similarity,
    reconstruction_metrics,
    retrieval_metrics,
    silhouette_metrics,
    similarity_metrics,
)
from training.masking import random_correspondence


@dataclass(frozen=True, slots=True)
class Evaluation:
    """What one pass over a split made of its latent space.

    Attributes:
        metrics: Every measured number, keyed as it is logged.
        intervals: Half the 95% interval around each mean, zero for a count.
        similarity: The mean cosine similarity of every ordered pair of classes. (C, C)
        names: The classes, in the order the matrix holds them.
        placed: Where each measured latent sits on the plane. (N, 2)
        classes: The class of each of them, in that same order.
    """

    metrics: dict[str, float]
    intervals: dict[str, float]
    similarity: np.ndarray
    names: list[str]
    placed: np.ndarray
    classes: list[str]


def pooled(grid: FeatureGrid) -> Tensor:
    """Return the one vector a feature's grid is read as, its occupied cells averaged.

    Args:
        grid: The grid every instrument the feature holds was read into.

    Returns:
        latent: The unit vector along the sum of its cells, zero where it holds none.
    """
    return functional.normalize(grid.values.sum(dim=1), dim=-1)  # (B, D)


def evaluate_latent_space(
    model: CrossSensorMAE, loader: DataLoader, config: Config, device: torch.device
) -> Evaluation:
    """Return what the model's latent space makes of one split's classes.

    Args:
        model: The model, loaded from a checkpoint and on the device.
        loader: The split to read, in batches, each feature read whole.
        config: How many neighbours a retrieval reads, and the layout seed.
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
            embedded.append(pooled(grid).cpu())  # (B, D)
            read += [one for one, _ in identities]
    latents = torch.cat(embedded)  # (N, D)
    # A feature holding no patch of any instrument the model reads embeds to nothing.
    kept = latents.norm(dim=-1) > 0  # (N,)
    classes = [one for one, was in zip(read, kept.tolist(), strict=True) if was]
    latents = latents[kept]  # (N, D)
    similarity, names = class_similarity(latents, classes)  # (C, C)
    # Evaluate retrieval, silhouette and similarity of the representations
    measured = (
        retrieval_metrics(latents, classes, config.evaluation.neighbours)
        | silhouette_metrics(latents, classes)
        | similarity_metrics(similarity)
    )
    # Package the metrics, their confidence bounds, and the 2D UMAP layout
    return Evaluation(
        metrics={name: value for name, (value, _) in measured.items()}
        | {
            "features": len(classes),
            "classes": len(names),
            "empty": len(read) - len(classes),
        },
        intervals={name: half for name, (_, half) in measured.items()},
        similarity=similarity,
        names=names,
        # Map normalized latent vectors into 2D space using cosine distance UMAP
        placed=UMAP(
            n_components=2, metric="cosine", random_state=config.dataset.seed
        ).fit_transform(latents.numpy()),  # (N, 2)
        classes=classes,
    )


def evaluate_reconstruction(
    model: CrossSensorMAE,
    loader: DataLoader,
    mask_ratio: float,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    """Return how well the model rebuilds the patches it was never shown."""
    model.eval()  # Switch model to evaluation mode
    generator = torch.Generator(device=device).manual_seed(
        seed
    )  # Seed generator for reproducible masking
    totals: defaultdict[str, float] = defaultdict(
        float
    )  # Accumulate metric totals across batches
    batches = 0
    with (
        torch.no_grad()
    ):  # Disable autograd to reduce memory usage and speed up execution
        for batch, cells, _ in loader:
            # Transfer input batch tensors to execution device
            batch = {name: tokens.to(device) for name, tokens in batch.items()}
            # Apply deterministic sensor masking pattern
            batch = random_correspondence(batch, mask_ratio, generator)
            # Generate patch reconstructions across all sensor pairings
            reconstruction = model(batch, cells.to(device))
            others = max(
                len(batch) - 1, 1
            )  # Count available cross-modal sources for averaging
            for asked, tokens in batch.items():
                # Identify valid patches that were masked out
                hidden = tokens.present & ~tokens.visible  # (B, K)
                for read in batch:
                    # Check the input sensor has visible context tokens
                    readable = batch[read].visible.any(dim=1, keepdim=True)  # (B, 1)
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
