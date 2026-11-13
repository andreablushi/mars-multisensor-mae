"""Training every protocol's model on the same batches, kept at its best validation."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import chain, repeat

import numpy as np
import torch
from mars_multisensor_dataset.analysis.ground_truth.swim.sample import (
    GRID_SPACING_KM,
    METRES_PER_KM,
)
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from wandb.sdk.wandb_run import Run

from benchmarking.architecture.model import BenchmarkModel
from benchmarking.evaluation.metrics import rmse_and_r2
from benchmarking.finetune.loss import mse
from benchmarking.finetune.step import batch_step
from benchmarking.finetune.validate import validate
from dataset.loader import tile_loader
from dataset.store import DatasetBuild
from logs.benchmark import log_epoch, log_step
from logs.tracker import console_logger

log = console_logger(__name__)

REACH_M = GRID_SPACING_KM * METRES_PER_KM / 2


def finetune(
    models: Mapping[str, BenchmarkModel],
    build: DatasetBuild,
    training: Sequence[str],
    validation: Sequence[str],
    targets: Mapping[str, np.ndarray],
    spread: np.ndarray,
    sizes: Mapping[str, Mapping[str, int]],
    shapes: Mapping[str, tuple[int, ...]],
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
        sizes: How far a patch of each instrument read runs along each axis it is
            cut on.
        shapes: The shape of one patch of each instrument as the model reads it.
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
            sizes,
            shapes,
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

    def lambda_lr_schedule(step: int) -> float:
        """Return the share of the peak rate one step runs at."""
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(num_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lambda_lr_schedule)
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
        # A point is scored where a patch the tile was cut to stands near it
        spots = torch.cat([one.position[..., :2] for one in batch.values()], 1)
        read = torch.cat([one.present for one in batch.values()], 1)  # (B, S)
        apart = torch.cdist(points.expand(len(spots), -1, -1), spots)  # (B, P * P, S)
        scored = apart.masked_fill(~read[:, None], math.inf).amin(-1) <= REACH_M
        losses = {
            protocol: mse(prediction, target, scale, scored.to(device))
            for protocol, prediction in predictions.items()
        }
        sum(losses.values()).backward()
        for model in models.values():
            clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
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
