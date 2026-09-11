"""Measuring what a trained model's latent space made of the classes it never read."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader

from architecture.mae import CrossSensorMAE
from config.schema import Config
from evaluation.embed import embed_split
from evaluation.projection import projected_latent
from evaluation.retrieval import retrieval_metrics
from evaluation.separability import (
    class_similarity,
    silhouette_metrics,
    similarity_metrics,
)


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
            each latent sits on the plane.

    Raises:
        ValueError: When fewer than two classes were read, which no metric here
            says anything about.
    """
    latents, read = embed_split(model, loader, device)  # (N, D)
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
        placed=projected_latent(latents, config.dataset.seed),  # (N, 2)
        classes=classes,
    )
