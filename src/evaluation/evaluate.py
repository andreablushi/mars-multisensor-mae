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
        similarity: The mean cosine similarity of every ordered pair of
            classes. (C, C)
        names: The classes, in the order the matrix holds them.
        placed: Where each measured latent sits on the plane. (N, 2)
        classes: The class of each of them, in that same order.
    """

    metrics: dict[str, float]
    similarity: np.ndarray
    names: list[str]
    placed: np.ndarray
    classes: list[str]


def evaluate_latent_space(
    model: CrossSensorMAE, loader: DataLoader, config: Config, device: torch.device
) -> Evaluation:
    """Return what the model's latent space makes of one split's classes.

    Args:
        model: The model, loaded from a checkpoint and on the device.
        loader: The split to read, in batches.
        config: How many neighbours a retrieval reads, and the seed the layout
            is fixed by.
        device: Where the model runs.

    Returns:
        evaluation: Every metric, the class similarities behind them, and where
            each latent sits on the plane. A feature is embedded over every
            patch it holds of every instrument, none hidden.

    Raises:
        ValueError: When fewer than two classes were read, which no metric here
            says anything about.
    """
    model.eval()
    embedded: list[Tensor] = []
    read: list[str] = []
    with torch.no_grad():
        for batch, feature_classes in loader:
            batch = {name: tokens.to(device) for name, tokens in batch.items()}
            embedded.append(model.embed(batch).cpu())  # (B, D)
            read.extend(feature_classes)
    latents = torch.cat(embedded)  # (N, D)
    # A feature holding no patch of any instrument the model reads embeds to nothing.
    counted = latents.norm(dim=-1) > 0  # (N,)
    classes = [one for one, held in zip(read, counted.tolist(), strict=True) if held]
    latents = functional.normalize(latents[counted], dim=-1)  # (N, D)
    if len(set(classes)) < 2:
        raise ValueError(f"{len(set(classes))} classes read, two say the least")
    similarity, names = class_similarity(latents, classes)  # (C, C)
    return Evaluation(
        metrics=retrieval_metrics(latents, classes, config.evaluation.neighbours)
        | silhouette_metrics(latents, classes)
        | similarity_metrics(similarity)
        | {
            "features": len(classes),
            "classes": len(names),
            "empty": len(read) - len(classes),
        },
        similarity=similarity,
        names=names,
        # The layout reads the cosine the metrics do, so a class held together draws so.
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
    """Return how well the model rebuilds the patches it was never shown.

    Args:
        model: The model, which is switched to evaluation.
        loader: The split to read, in batches.
        mask_ratio: The share of each instrument's patches hidden from it.
        seed: What the mask is drawn from, so two passes hide the same patches.
        device: Where the model runs.

    Returns:
        metrics: Under "umr/<instrument>/<metric>" what an instrument rebuilt of
            its own hidden patches, and under "cmr/<instrument>/<metric>" what
            every other instrument rebuilt of those same patches, averaged over
            them. Each metric is what reconstruction_metrics names it, averaged
            over the batches.
    """
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    totals: defaultdict[str, float] = defaultdict(float)
    batches = 0
    with torch.no_grad():
        for batch, _ in loader:
            batch = {name: tokens.to(device) for name, tokens in batch.items()}
            batch = random_correspondence(batch, mask_ratio, generator)
            reconstruction = model(batch)
            others = max(len(batch) - 1, 1)
            for asked, tokens in batch.items():
                hidden = tokens.present & ~tokens.visible  # (B, K)
                for read in batch:
                    readable = batch[read].visible.any(dim=1, keepdim=True)  # (B, 1)
                    own = read == asked
                    measured = reconstruction_metrics(
                        reconstruction.predictions[asked, read],
                        tokens.values,
                        tokens.valid,
                        hidden if own else hidden & readable,
                    )
                    for name, value in measured.items():
                        key = f"{'umr' if own else 'cmr'}/{asked}/{name}"
                        totals[key] += value if own else value / others
            batches += 1
    return {name: value / max(batches, 1) for name, value in totals.items()}
