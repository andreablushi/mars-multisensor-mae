"""The optimizer and learning rate schedule each protocol's model steps with."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from functools import partial

import torch
from torch.optim.lr_scheduler import LambdaLR

from benchmarking.architecture.model import BenchmarkModel
from training.optimizer import cosine_warmup_share


def finetune_optimizers_and_schedules(
    models: Mapping[str, BenchmarkModel],
    head_learning_rate: float,
    encoder_learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    num_steps: int,
) -> dict[str, tuple[torch.optim.AdamW, LambdaLR]]:
    """Return one AdamW per model over its trained weights, grouped by rate and decay.

    Args:
        models: Each protocol's model, its frozen weights left out.
        head_learning_rate: The probes' peak learning rate.
        encoder_learning_rate: The encoders' peak learning rate.
        weight_decay: The AdamW weight decay of every matrix.
        warmup_steps: How many steps the rate climbs before the cosine decay.
        num_steps: How many steps the schedule runs for.

    Returns:
        optimizers: Each protocol's AdamW, the probe at the head rate and the rest
            at the encoder rate, vectors undecayed, with the warmup and cosine
            schedule its rates follow, keyed by protocol.
    """
    optimizers = {}
    for protocol, model in models.items():
        groups = defaultdict(list)
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
        optimizers[protocol] = (
            optimizer,
            LambdaLR(
                optimizer,
                partial(
                    cosine_warmup_share, warmup_steps=warmup_steps, num_steps=num_steps
                ),
            ),
        )
    return optimizers
