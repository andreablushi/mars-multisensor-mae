"""Training a head until its validation stops improving, the model frozen or not."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from wandb.sdk.wandb_run import Run

from architecture.mae import CrossSensorMAE
from benchmarking.evaluation.metrics import ice_scores
from benchmarking.finetune.loss import ice_loss
from benchmarking.finetune.step import batch_logits
from benchmarking.finetune.validate import predicted_probabilities
from dataset.loader import tile_loader
from dataset.store import DatasetBuild
from logs.benchmark import log_epoch
from logs.tracker import console_logger

log = console_logger(__name__)


def finetune(
    model: CrossSensorMAE,
    head: nn.Module,
    build: DatasetBuild,
    training: Sequence[str],
    validation: Sequence[str],
    targets: Mapping[str, np.ndarray],
    segmented: bool,
    tuned: bool,
    sizes: Mapping[str, Mapping[str, int]],
    shapes: Mapping[str, tuple[int, ...]],
    budget: Mapping[str, int],
    budget_scale: float,
    batch_size: int,
    whole_batch_size: int,
    workers: int,
    epochs: int,
    patience: int,
    warmup_epochs: int,
    head_learning_rate: float,
    encoder_learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    run: Run,
    prefix: str,
) -> None:
    """Train a head on the model's tokens, both left as its best validation epoch.

    Args:
        model: The model, already on the device.
        head: The classifier or the segmenter, already on the device.
        build: The build the tiles are read from.
        training: The training tiles.
        validation: The validation tiles.
        targets: Whether SWIM finds ice at each depth, as 0 or 1, keyed by the tile.
            (Z,) or (P * P, Z)
        segmented: Whether the head scores every point, rather than the tile.
        tuned: Whether the encoders train with the head, or stay frozen.
        sizes: How far a patch of each sensor read runs along each axis it is cut on.
        shapes: The shape of one patch of each instrument as the model reads it.
        budget: How many patches of each instrument a training tile keeps.
        budget_scale: The most a training batch's budget is multiplied by.
        batch_size: How many tiles one training batch holds.
        whole_batch_size: How many tiles one validation batch holds, each read whole.
        workers: How many processes read tiles beside the training.
        epochs: The most epochs the head trains for.
        patience: How many epochs without a better validation AUPRC stop it.
        warmup_epochs: How many epochs the rate climbs before the cosine decay.
        head_learning_rate: The head's peak learning rate.
        encoder_learning_rate: The encoders' peak learning rate when tuned.
        weight_decay: The AdamW weight decay.
        seed: What fixes the order, the readers' own draws and each fixed cut.
        device: Where the model runs.
        run: The tracked run every epoch is logged to.
        prefix: Where the head's epochs are logged.
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
        )
        for tiles, cut, size, shuffled in (
            (training, budget, batch_size, True),
            (validation, None, whole_batch_size, False),
        )
    )
    model.requires_grad_(tuned)
    heads = list(head.parameters())
    encoders = [*model.encoders.parameters(), *model.crossencoder.parameters()]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [one for one in heads if one.ndim > 1],
                "lr": head_learning_rate,
            },
            {
                "params": [one for one in heads if one.ndim <= 1],
                "lr": head_learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": [one for one in encoders if one.ndim > 1],
                "lr": encoder_learning_rate,
            },
            {
                "params": [one for one in encoders if one.ndim <= 1],
                "lr": encoder_learning_rate,
                "weight_decay": 0.0,
            },
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
    best, waited, kept = -math.inf, 0, None
    model.train()
    head.train()
    for epoch in range(1, epochs + 1):
        losses = []
        for batch, identities in training:
            optimizer.zero_grad()
            truth = np.stack([targets[tile] for tile in identities])
            loss = ice_loss(
                batch_logits(model, head, batch, device),
                torch.as_tensor(truth, device=device),
            )
            loss.backward()
            clip_grad_norm_([*heads, *encoders], 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.detach())
        scores = ice_scores(
            predicted_probabilities(model, head, validation, device),
            targets,
            segmented,
        )
        log_epoch(run, prefix, epoch, float(torch.stack(losses).mean()), scores)
        gained = float(np.mean([held["AUPRC"] for held in scores.values()]))
        log.info("%s epoch %d validation AUPRC %.4f", prefix, epoch, gained)
        if gained > best:
            best, waited = gained, 0
            kept = deepcopy((head.state_dict(), model.state_dict()))
        else:
            waited += 1
        if waited == patience:
            break
    head.load_state_dict(kept[0])
    model.load_state_dict(kept[1])
