"""Training the model over one split for a fixed schedule, checking it on another."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from itertools import chain, repeat
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from wandb.sdk.wandb_run import Run

from mars_multisensor_mae.architecture.mae import CrossSensorMAE
from mars_multisensor_mae.architecture.models.context import Pooling
from mars_multisensor_mae.configs.load import mask_generators
from mars_multisensor_mae.configs.paths import checkpoint_path
from mars_multisensor_mae.dataset.loader import tile_loader
from mars_multisensor_mae.dataset.models.patch import PatchLayout
from mars_multisensor_mae.dataset.store import DatasetBuild
from mars_multisensor_mae.logs.tracker import (
    TRAIN,
    VALIDATION,
    console_logger,
    log_terms,
)
from mars_multisensor_mae.training.optimizer import optimizer_and_schedule
from mars_multisensor_mae.training.step import autocast_dtype, batch_step
from mars_multisensor_mae.training.validate import validation_terms

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
    max_grad_norm: float,
    mask_ratio: float,
    aligned: Sequence[str],
    pooling: Pooling,
    temperature: float,
    latent_temperature: float,
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
        max_grad_norm: The norm every step's gradient is clipped to.
        mask_ratio: The share of each instrument's patches hidden.
        aligned: The instruments MIM aligns, none for no MIM.
        pooling: What MIM aligns of each instrument, its GAP or its CLS.
        temperature: What MIM divides every cosine by.
        latent_temperature: What latent UMR and CMR divide every cosine by.
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
    optimizer, scheduler = optimizer_and_schedule(
        model, learning_rate, weight_decay, warmup_steps, num_steps
    )
    scaler = torch.amp.GradScaler(
        device.type, enabled=autocast_dtype(device) == torch.float16
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
            terms = batch_step(
                model,
                batch,
                mask_ratio,
                aligned,
                pooling,
                temperature,
                latent_temperature,
                generators,
                device,
            )
            scaler.scale(terms["loss"]).backward()
            scaler.unscale_(optimizer)
            # Stabilize training against exploding gradients
            norm = clip_grad_norm_(model.parameters(), max_grad_norm)
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
            model,
            validation,
            mask_ratio,
            aligned,
            pooling,
            temperature,
            latent_temperature,
            mask_generators(model.encoders, seed, device),
            device,
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
