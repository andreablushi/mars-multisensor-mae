"""Training the heads on one read of the model's tokens, the model frozen or adapted."""

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
from benchmarking.finetune.loss import ice_losses
from benchmarking.finetune.step import batch_consistency
from benchmarking.finetune.validate import predicted_consistency
from dataset.loader import tile_loader
from dataset.store import DatasetBuild
from logs.benchmark import log_epoch, log_step
from logs.tracker import console_logger

log = console_logger(__name__)


def finetune(
    model: CrossSensorMAE,
    heads: Mapping[str, nn.Module],
    build: DatasetBuild,
    training: Sequence[str],
    validation: Sequence[str],
    tested: DataLoader,
    targets: Mapping[str, Mapping[str, np.ndarray]],
    icy: Mapping[str, Mapping[str, np.ndarray]],
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
) -> dict[str, dict[str, np.ndarray]]:
    """Train every head together, then return each one's test consistency at its best.

    Args:
        model: The model, on the device, only its parameters asking for gradients
            trained.
        heads: The classifier and the segmenter, on the device, keyed by task.
        build: The build the tiles are read from.
        training: The training tiles.
        validation: The validation tiles.
        tested: The test tiles, in batches, each read whole.
        targets: SWIM's consistency at each depth, the tile's median or every
            point's, keyed by task, then by tile. (Z,) or (P * P, Z)
        icy: Whether SWIM finds ice there, keyed by task, then by tile. (Z,) or
            (P * P, Z)
        sizes: How far a patch of each sensor read runs along each axis it is cut on.
        shapes: The shape of one patch of each instrument as the model reads it.
        budget: How many patches of each instrument a training tile keeps.
        budget_scale: The most a training batch's budget is multiplied by.
        batch_size: How many tiles one training batch holds.
        whole_batch_size: How many tiles one validation batch holds, each read whole.
        workers: How many processes read tiles beside the training.
        epochs: How many epochs the heads train for.
        warmup_epochs: How many epochs the rate climbs before the cosine decay.
        head_learning_rate: The heads' peak learning rate.
        model_learning_rate: The peak learning rate of the model's trained
            parameters.
        weight_decay: The AdamW weight decay.
        max_grad_norm: The norm every step's gradient is clipped to.
        seed: What fixes the order, the readers' own draws and each fixed cut.
        device: Where the model runs.
        run: The tracked run every step is logged to.
        prefix: Where the protocol's steps are logged.

    Returns:
        consistency: Each head's test consistency at its best validation epoch,
            keyed by task, then by tile. (Z,) or (P * P, Z)
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
        (
            [one for head in heads.values() for one in head.parameters()],
            head_learning_rate,
        ),
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
    best = dict.fromkeys(heads, -math.inf)
    kept, step = {}, 0
    model.train()
    for head in heads.values():
        head.train()
    for epoch in range(1, epochs + 1):
        for batch, identities in training:
            optimizer.zero_grad()
            consistency, covered = batch_consistency(model, heads, batch, device)
            losses = ice_losses(
                consistency,
                torch.as_tensor(
                    np.stack([targets["segmentation"][tile] for tile in identities]),
                    device=device,
                ),
                covered,
            )
            sum(losses.values()).backward()
            clip_grad_norm_(
                [one for group in optimizer.param_groups for one in group["params"]],
                max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            step += 1
            log_step(
                run, prefix, step, {task: float(one) for task, one in losses.items()}
            )
        predicted = predicted_consistency(model, heads, validation, device)
        scores = {
            task: ice_scores(held, targets[task], icy[task], task == "segmentation")
            for task, held in predicted.items()
        }
        log_epoch(run, prefix, step, scores)
        for task, by_depth in scores.items():
            gained = float(np.mean([held["AUPRC"] for held in by_depth.values()]))
            log.info(
                "%s/%s epoch %d validation AUPRC %.4f", prefix, task, epoch, gained
            )
            if gained > best[task]:
                best[task] = gained
                kept[task] = deepcopy(
                    (
                        heads[task].state_dict(),
                        {name: one.detach() for name, one in trained.items()},
                    )
                )
    consistency = {}
    for task, (head, parameters) in kept.items():
        heads[task].load_state_dict(head)
        model.load_state_dict(parameters, strict=False)
        consistency[task] = predicted_consistency(
            model, {task: heads[task]}, tested, device
        )[task]
    return consistency
