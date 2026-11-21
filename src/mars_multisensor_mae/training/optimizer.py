"""The optimizer, the learning rate schedule and the target encoder's EMA."""

from __future__ import annotations

import math
from collections.abc import Sequence
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


def ema_momentum(step: int, num_steps: int, momentum_range: Sequence[float]) -> float:
    """Return I-JEPA's EMA momentum at one step, linear from its first to last value.

    Args:
        step: The step, from 1.
        num_steps: How many steps the run takes.
        momentum_range: The momentum at the start and at the end.

    Returns:
        momentum: The share of the target encoder's weights kept at this step.
    """
    first, last = momentum_range
    return first + (last - first) * step / num_steps


def update_target_encoder(
    target: CrossSensorMAE, model: CrossSensorMAE, momentum: float
) -> None:
    """Move every target encoder weight towards the model's, keeping a share of its own.

    Args:
        target: The target encoder, updated in place.
        model: The model whose encoders it follows.
        momentum: The share of each target weight kept.
    """
    online = dict(model.named_parameters())
    with torch.no_grad():
        held = dict(target.named_parameters())
        torch._foreach_lerp_(
            list(held.values()), [online[name] for name in held], 1 - momentum
        )
