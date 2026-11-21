"""The optimizer and learning rate schedule pretraining steps with."""

from __future__ import annotations

import math
from functools import partial

import torch
from torch.optim.lr_scheduler import LambdaLR

from mars_multisensor_mae.architecture.mae import CrossSensorMAE


def cosine_warmup_share(step: int, warmup_steps: int, num_steps: int) -> float:
    """Return the share of the peak rate one step runs at, a linear warmup then cosine.

    Args:
        step: The step, from 0.
        warmup_steps: How many steps the rate climbs before the cosine decay.
        num_steps: How many steps the schedule runs for.

    Returns:
        share: The step's rate over the peak rate.
    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(num_steps - warmup_steps, 1)
    return 0.5 * (1 + math.cos(math.pi * progress))


def optimizer_and_schedule(
    model: CrossSensorMAE,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    num_steps: int,
) -> tuple[torch.optim.AdamW, LambdaLR]:
    """Return AdamW over the model, its vectors undecayed, and the rate's schedule.

    Args:
        model: The model trained.
        learning_rate: The peak learning rate.
        weight_decay: The AdamW weight decay of every matrix.
        warmup_steps: How many steps the rate climbs before the cosine decay.
        num_steps: How many steps the schedule runs for.

    Returns:
        optimizer: AdamW with MAE's betas, matrices decayed and the rest not.
        scheduler: The warmup and cosine decay the optimizer's rate follows.
    """
    # Biases, norms and learned vectors are not decayed, as MAE leaves them
    decayed = [one for one in model.parameters() if one.ndim > 1]
    kept = [one for one in model.parameters() if one.ndim <= 1]
    optimizer = torch.optim.AdamW(
        [{"params": decayed}, {"params": kept, "weight_decay": 0.0}],
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        fused=True,
    )
    return optimizer, LambdaLR(
        optimizer,
        partial(cosine_warmup_share, warmup_steps=warmup_steps, num_steps=num_steps),
    )
