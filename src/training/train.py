"""Training the model over one split for a fixed schedule, checking it on another."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from functools import partial
from itertools import chain, repeat
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from wandb.sdk.wandb_run import Run

from architecture.mae import CrossSensorMAE
from architecture.models.context import Pooling
from configs.load import mask_generators
from configs.paths import checkpoint_path
from dataset.loader import tile_loader
from dataset.models.patch import PatchLayout
from dataset.store import DatasetBuild
from logs.tracker import TRAIN, VALIDATION, console_logger, log_terms
from training.optimizer import pretraining_optimizer_and_schedule
from training.step import autocast_dtype, batch_step
from training.validate import validation_terms

log = console_logger(__name__)


def train(
    model: CrossSensorMAE,
    build: DatasetBuild,
    training: Sequence[str],
    validation: Sequence[str],
    layout: PatchLayout,
    budget: Mapping[str, int],
    budget_scale: float,
    batch_size: int,
    workers: int,
    num_epochs: int,
    steps_per_epoch: int,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    mask_ratio: float,
    aligned: Sequence[str],
    pooling: Pooling,
    temperature: float,
    checkpoints: str,
    seed: int,
    device: torch.device,
    run: Run,
    config: Mapping[str, object],
) -> Path:
    """Return where the last checkpoint was written, after training the model.

    Args:
        model: The model, already on the device.
        build: The build the tiles are read from.
        training: The training tiles.
        validation: The validation tiles.
        layout: How each instrument is cut into patches, and the shape the model
            reads one in.
        budget: How many patches of each instrument a tile keeps.
        budget_scale: The most a training batch's budget is multiplied by.
        batch_size: How many tiles one batch holds.
        workers: How many processes read tiles beside the training.
        num_epochs: How many epochs the run takes, a validation after each.
        steps_per_epoch: How many steps one epoch takes.
        learning_rate: The peak learning rate, reached after the warmup.
        weight_decay: The AdamW weight decay.
        warmup_steps: How many steps the rate climbs before the cosine decay.
        mask_ratio: The share of each instrument's patches hidden.
        aligned: The instruments MIM aligns, none for no MIM.
        pooling: What MIM aligns of each instrument, its GAP or its CLS.
        temperature: What MIM divides every cosine by.
        checkpoints: Where checkpoints are written, relative to the repository.
        seed: What fixes the masks, the order and the validation cuts.
        device: Where the model runs.
        run: The tracked run every metric is logged to.
        config: The run's whole config, saved beside the weights.

    Returns:
        last: The checkpoint the run ended on.
    """
    training, validation = (
        tile_loader(
            build,
            tiles,
            layout,
            budget,
            budget_scale,
            seed,
            batch_size,
            workers,
            shuffled,
        )
        for tiles, shuffled in ((training, True), (validation, False))
    )
    num_steps = num_epochs * steps_per_epoch
    optimizer, scheduler = pretraining_optimizer_and_schedule(
        model, learning_rate, weight_decay, warmup_steps, num_steps
    )
    scaler = torch.amp.GradScaler(
        device.type, enabled=autocast_dtype(device) == torch.float16
    )
    step_terms = partial(
        batch_step,
        model,
        mask_ratio=mask_ratio,
        aligned=aligned,
        pooling=pooling,
        temperature=temperature,
        device=device,
    )
    generators = mask_generators(model.encoders, seed, device)
    started = time.perf_counter()
    batches = chain.from_iterable(repeat(training))
    for epoch in range(num_epochs):
        model.train()
        for step in range(
            epoch * steps_per_epoch + 1, (epoch + 1) * steps_per_epoch + 1
        ):
            batch, identities = next(batches)
            optimizer.zero_grad()
            terms = step_terms(batch, generators=generators)
            scaler.scale(terms["loss"]).backward()
            scaler.unscale_(optimizer)
            # Stabilize training against exploding gradients
            norm = clip_grad_norm_(model.parameters(), 10.0)
            if not scaler.is_enabled() and not torch.isfinite(norm):
                raise FloatingPointError(
                    f"step {step} has a non-finite gradient: {identities}"
                )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            log_terms(run, step, TRAIN, terms | {"grad_norm": norm})
        model.eval()
        metrics = validation_terms(
            validation, step_terms, mask_generators(model.encoders, seed, device)
        )
        log_terms(run, step, VALIDATION, metrics)
        log.info("step %d validation loss %.4f", step, metrics["loss"])
    last = checkpoint_path(checkpoints, "last")
    last.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "step": num_steps,
            "config": dict(config),
        },
        last,
    )
    run.summary.update(
        {"steps": num_steps, "run_seconds": time.perf_counter() - started}
    )
    return last
