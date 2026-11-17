"""The optimizer and learning rate schedule every protocol's model steps with."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from functools import partial

import torch
from torch.optim.lr_scheduler import LambdaLR

from benchmarking.architecture.model import BenchmarkModel
from training.optimizer import cosine_warmup_share


def finetune_optimizer_and_schedule(
    models: Mapping[str, BenchmarkModel],
    head_learning_rate: float,
    encoder_learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    num_steps: int,
) -> tuple[torch.optim.AdamW, LambdaLR]:
    """Return one AdamW over every model's trained weights, grouped by rate and decay.

    Args:
        models: Each protocol's model, its frozen weights left out.
        head_learning_rate: The probes' peak learning rate.
        encoder_learning_rate: The encoders' peak learning rate.
        weight_decay: The AdamW weight decay of every matrix.
        warmup_steps: How many steps the rate climbs before the cosine decay.
        num_steps: How many steps the schedule runs for.

    Returns:
        optimizer: AdamW, probes at the head rate and the rest at the encoder rate,
            vectors undecayed.
        scheduler: The warmup and cosine decay every group's rate follows.
    """
    groups = defaultdict(list)
    for model in models.values():
        for name, one in model.named_parameters():
            if one.requires_grad:
                rate = (
                    head_learning_rate
                    if name.startswith("probe.")
                    else encoder_learning_rate
                )
                groups[rate, weight_decay if one.ndim > 1 else 0.0].append(one)
    optimizer = torch.optim.AdamW(
        [
            {"params": held, "lr": rate, "weight_decay": decay}
            for (rate, decay), held in groups.items()
        ]
    )
    return optimizer, LambdaLR(
        optimizer,
        partial(cosine_warmup_share, warmup_steps=warmup_steps, num_steps=num_steps),
    )
