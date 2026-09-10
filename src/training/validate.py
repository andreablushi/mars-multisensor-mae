"""Measuring a model over one split: its loss, and how it retrieves its features."""

from __future__ import annotations

from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from architecture.mae import CrossSensorMAE
from config.schema import Config
from evaluation.embed import embed_batch
from evaluation.metrics import retrieval_metrics
from training.loss import total_loss


def validate(
    model: CrossSensorMAE, loader: DataLoader, config: Config, device: torch.device
) -> dict[str, float]:
    """Return the loss terms over one split, and its retrieval metrics.

    Args:
        model: The model, which is switched to evaluation.
        loader: The split, in batches.
        config: How much uniformity weighs, and how many neighbours count.
        device: Where the model runs.

    Returns:
        metrics: Every loss term averaged over the batches, keyed as the loss
            names them, and every retrieval metric over the split's features,
            keyed as the metrics name them.
    """
    model.eval()
    totals = defaultdict(float)
    batches = 0
    embeddings, labels = [], []
    with torch.no_grad():
        for batch, classes in loader:
            batch = {name: tokens.to(device) for name, tokens in batch.items()}
            terms = total_loss(model(batch), batch, config.training.uniformity_weight)
            for name, value in terms.items():
                totals[name] += float(value)
            batches += 1
            embeddings.append(embed_batch(model, batch))
            labels.extend(classes)
    metrics = {name: value / max(batches, 1) for name, value in totals.items()}
    return metrics | retrieval_metrics(
        torch.cat(embeddings), labels, config.training.neighbours
    )
