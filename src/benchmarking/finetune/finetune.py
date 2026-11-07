"""Training the map head over the model's tokens, the model frozen or adapted."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from wandb.sdk.wandb_run import Run

from architecture.mae import CrossSensorMAE
from benchmarking.evaluation.metrics import ice_scores
from benchmarking.finetune.step import batch_consistency
from benchmarking.finetune.validate import predicted_consistency
from dataset.loader import tile_loader
from dataset.store import DatasetBuild
from logs.benchmark import log_epoch, log_step
from logs.tracker import console_logger

log = console_logger(__name__)


def finetune(
    model: CrossSensorMAE,
    head: nn.Module,
    build: DatasetBuild,
    training: Sequence[str],
    validation: Sequence[str],
    tested: DataLoader,
    targets: Mapping[str, np.ndarray],
    sizes: Mapping[str, Mapping[str, int]],
    shapes: Mapping[str, tuple[int, ...]],
    budget: Mapping[str, int],
    budget_scale: float,
    batch_size: int,
    whole_batch_size: int,
    workers: int,
    epochs: int,
    warmup_epochs: int,
    head_learning_rate: float,
    model_learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    seed: int,
    device: torch.device,
    run: Run,
    prefix: str,
) -> dict[str, np.ndarray]:
    """Train the head, then return its test consistency at its best.

    Args:
        model: The model, on the device, only its parameters asking for gradients
            trained.
        head: The map head, on the device.
        build: The build the tiles are read from.
        training: The training tiles.
        validation: The validation tiles.
        tested: The test tiles, in batches, each read whole.
        targets: SWIM's consistency of every point at each depth, keyed by the
            tile. (P * P, Z)
        sizes: How far a patch of each sensor read runs along each axis it is cut on.
        shapes: The shape of one patch of each instrument as the model reads it.
        budget: How many patches of each instrument a training tile keeps.
        budget_scale: The most a training batch's budget is multiplied by.
        batch_size: How many tiles one training batch holds.
        whole_batch_size: How many tiles one validation batch holds, each read whole.
        workers: How many processes read tiles beside the training.
        epochs: How many epochs the head trains for.
        warmup_epochs: How many epochs the rate climbs before the cosine decay.
        head_learning_rate: The head's peak learning rate.
        model_learning_rate: The peak learning rate of the model's trained
            parameters.
        weight_decay: The AdamW weight decay.
        max_grad_norm: The norm every step's gradient is clipped to.
        seed: What fixes the order, the readers' own draws and each fixed cut.
        device: Where the model runs.
        run: The tracked run every step is logged to.
        prefix: Where the protocol's steps are logged.

    Returns:
        consistency: The test consistency at the best validation epoch, keyed by
            the tile. (P * P, Z)
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
            size,
            workers,
            shuffled,
            shuffled,
        )
        for tiles, cut, size, shuffled in (
            (training, budget, batch_size, True),
            (validation, None, whole_batch_size, False),
        )
    )
    trained = {name: one for name, one in model.named_parameters() if one.requires_grad}
    groups = (
        (list(head.parameters()), head_learning_rate),
        (list(trained.values()), model_learning_rate),
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": [one for one in parameters if one.ndim > 1], "lr": rate}
            for parameters, rate in groups
        ]
        + [
            {
                "params": [one for one in parameters if one.ndim <= 1],
                "lr": rate,
                "weight_decay": 0.0,
            }
            for parameters, rate in groups
        ],
        weight_decay=weight_decay,
    )
    num_steps, warmup_steps = epochs * len(training), warmup_epochs * len(training)

    def lambda_lr_schedule(step: int) -> float:
        """Return the share of the peak rate one step runs at."""
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(num_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lambda_lr_schedule)
    best, step = math.inf, 0
    model.train()
    head.train()
    for epoch in range(1, epochs + 1):
        for batch, identities in training:
            optimizer.zero_grad()
            consistency, covered = batch_consistency(model, head, batch, device)
            wanted = torch.as_tensor(
                np.stack([targets[tile] for tile in identities]), device=device
            )
            error = (consistency - wanted) ** 2 * covered[..., None]
            loss = error.sum() / (covered.sum() * wanted.shape[-1])
            loss.backward()
            clip_grad_norm_(
                [one for group in optimizer.param_groups for one in group["params"]],
                max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            step += 1
            log_step(run, prefix, step, float(loss.detach()))
        scores = ice_scores(
            predicted_consistency(model, head, validation, device), targets
        )
        log_epoch(run, prefix, step, scores)
        rmse = float(np.mean([held["RMSE"] for held in scores.values()]))
        log.info("%s epoch %d validation RMSE %.4f", prefix, epoch, rmse)
        if rmse < best:
            best = rmse
            kept = deepcopy(
                (
                    head.state_dict(),
                    {name: one.detach() for name, one in trained.items()},
                )
            )
    head.load_state_dict(kept[0])
    model.load_state_dict(kept[1], strict=False)
    return predicted_consistency(model, head, tested, device)
