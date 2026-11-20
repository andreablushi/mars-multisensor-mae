"""Training every protocol's model on the same batches, kept at its best validation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import chain, repeat

import numpy as np
import torch
from ice_consistency.sample import (
    GRID_SPACING_KM,
    METRES_PER_KM,
)
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from wandb.sdk.wandb_run import Run

from benchmarking.architecture.model import BenchmarkModel
from benchmarking.evaluation.metrics import rmse_and_r2
from benchmarking.finetune.loss import mse
from benchmarking.finetune.optimizer import finetune_optimizer_and_schedule
from benchmarking.finetune.step import batch_step, scored_swim_points
from benchmarking.finetune.validate import validate
from dataset.loader import tile_loader
from dataset.models.patch import PatchLayout
from dataset.store import DatasetBuild
from logs.benchmark import log_epoch, log_step
from logs.tracker import console_logger
from training.step import autocast_dtype

log = console_logger(__name__)

REACH_M = GRID_SPACING_KM * METRES_PER_KM / 2


def finetune(
    models: Mapping[str, BenchmarkModel],
    build: DatasetBuild,
    training: Sequence[str],
    validation: Sequence[str],
    targets: Mapping[str, np.ndarray],
    spread: np.ndarray,
    layout: PatchLayout,
    budget: Mapping[str, int],
    budget_scale: float,
    training_batch_size: int,
    batch_size: int,
    workers: int,
    num_steps: int,
    warmup_steps: int,
    validate_every: int,
    head_learning_rate: float,
    encoder_learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    seed: int,
    device: torch.device,
    run: Run,
    share: str,
) -> dict[str, dict[str, Tensor]]:
    """Return every model's trained weights, each at its best validation.

    Args:
        models: Each protocol's model, on the device, only its parameters asking
            for gradients trained.
        build: The build the tiles are read from.
        training: The training tiles.
        validation: The validation tiles.
        targets: SWIM's value of every point for each target, keyed by the
            tile. (P * P, Z)
        spread: Each target's standard deviation, which the loss divides by. (Z,)
        layout: How each instrument read is cut into patches, and the shape the
            model reads one in.
        budget: How many patches of each instrument a training tile keeps.
        budget_scale: The most a training batch's budget is multiplied by.
        training_batch_size: How many tiles one training batch holds, each cut to
            the budget around one centre.
        batch_size: How many tiles one validation batch holds, each read whole.
        workers: How many processes read tiles beside the training.
        num_steps: How many steps the models train for.
        warmup_steps: How many steps the rate climbs before the cosine decay.
        validate_every: How many steps between two validations.
        head_learning_rate: The probes' peak learning rate.
        encoder_learning_rate: The encoders' peak learning rate.
        weight_decay: The AdamW weight decay.
        max_grad_norm: The norm each model's gradient is clipped to every step.
        seed: What fixes the order of the training tiles.
        device: Where the models run.
        run: The tracked run every step is logged to.
        share: What the share of the tiles trained on is logged as.

    Returns:
        kept: Each model's trained parameters at its highest mean validation R2,
            keyed by protocol, then by parameter.
    """
    training, validation = (
        tile_loader(
            build,
            tiles,
            layout,
            cut,
            budget_scale,
            seed,
            held,
            workers,
            shuffled,
            shuffled,
        )
        for tiles, cut, held, shuffled in (
            (training, budget, training_batch_size, True),
            (validation, None, batch_size, False),
        )
    )
    optimizer, scheduler = finetune_optimizer_and_schedule(
        models,
        head_learning_rate,
        encoder_learning_rate,
        weight_decay,
        warmup_steps,
        num_steps,
    )
    scaler = torch.amp.GradScaler(
        device.type, enabled=autocast_dtype(device) == torch.float16
    )
    scale = torch.as_tensor(spread, device=device)
    best, kept = dict.fromkeys(models, -math.inf), {}
    points = next(iter(models.values())).probe.points[:, :2].cpu()  # (P * P, 2)
    batches = chain.from_iterable(repeat(training))
    for step in range(1, num_steps + 1):
        batch, identities = next(batches)
        optimizer.zero_grad()
        predictions = batch_step(models, batch, device)
        target = torch.as_tensor(
            np.stack([targets[tile] for tile in identities]), device=device
        )
        scored = scored_swim_points(batch, points, REACH_M).to(device)
        losses = {
            protocol: mse(prediction, target, scale, scored)
            for protocol, prediction in predictions.items()
        }
        scaler.scale(sum(losses.values())).backward()
        scaler.unscale_(optimizer)
        for model in models.values():
            clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        log_step(run, step, share, {name: float(one) for name, one in losses.items()})
        if step % validate_every and step < num_steps:
            continue
        scores = rmse_and_r2(validate(models, validation, device), targets)
        log_epoch(run, step, share, scores)
        for protocol, model in models.items():
            fit = float(np.mean([held["R2"] for held in scores[protocol].values()]))
            log.info("%s %s step %d validation R2 %.4f", protocol, share, step, fit)
            if fit > best[protocol]:
                best[protocol] = fit
                kept[protocol] = {
                    name: one.detach().cpu().clone()
                    for name, one in model.named_parameters()
                    if one.requires_grad
                }
    return kept
