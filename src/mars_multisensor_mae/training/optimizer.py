"""The optimizer, the learning rate schedule and the target encoder's EMA."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from functools import partial

import torch
from torch import nn
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


def decayed_parameter_groups(
    parameters: Iterable[nn.Parameter], learning_rate: float, weight_decay: float
) -> list[dict[str, object]]:
    """Return AdamW's groups over some parameters at one rate, matrices alone decayed.

    Args:
        parameters: The parameters trained at this rate.
        learning_rate: Their peak learning rate.
        weight_decay: The AdamW weight decay of every matrix.

    Returns:
        groups: The matrices and the rest, each a group when it holds a parameter.
    """
    # Biases, norms and learned vectors are not decayed, as MAE leaves them
    held = list(parameters)
    groups = [
        {
            "params": [one for one in held if one.ndim > 1],
            "lr": learning_rate,
            "weight_decay": weight_decay,
        },
        {
            "params": [one for one in held if one.ndim <= 1],
            "lr": learning_rate,
            "weight_decay": 0.0,
        },
    ]
    return [group for group in groups if group["params"]]


def grouped_optimizer_and_schedule(
    groups: Sequence[Mapping[str, object]],
    betas: tuple[float, float],
    fused: bool | None,
    warmup_steps: int,
    num_steps: int,
) -> tuple[torch.optim.AdamW, LambdaLR]:
    """Return AdamW over some parameter groups, and the schedule all their rates follow.

    Args:
        groups: Each group's parameters, peak learning rate and weight decay.
        betas: AdamW's moment decays.
        fused: Whether AdamW runs its fused kernel, or None for torch's choice.
        warmup_steps: How many steps the rate climbs before the cosine decay.
        num_steps: How many steps the schedule runs for.

    Returns:
        optimizer: AdamW over the groups.
        scheduler: The warmup and cosine decay every group's rate follows.
    """
    optimizer = torch.optim.AdamW(
        [dict(group) for group in groups], betas=betas, fused=fused
    )
    return optimizer, LambdaLR(
        optimizer,
        partial(cosine_warmup_share, warmup_steps=warmup_steps, num_steps=num_steps),
    )


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
    return grouped_optimizer_and_schedule(
        decayed_parameter_groups(model.parameters(), learning_rate, weight_decay),
        (0.9, 0.95),
        True,
        warmup_steps,
        num_steps,
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
