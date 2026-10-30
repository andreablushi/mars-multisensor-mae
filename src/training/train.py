"""Training the model over one split for a fixed schedule, checking it on another."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from itertools import chain, repeat
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from wandb.sdk.wandb_run import Run

from architecture.mae import CrossSensorMAE
from configs.paths import checkpoint_path
from dataset.loader import tile_loader
from dataset.store import DatasetBuild
from logs.console import console_logger
from logs.tracker import TRAIN, VALIDATION, log_terms
from training.step import mask_generators, step_terms
from training.validate import validation_terms

log = console_logger(__name__)


def train(
    model: CrossSensorMAE,
    build: DatasetBuild,
    training: Sequence[str],
    validation: Sequence[str],
    sizes: Mapping[str, Mapping[str, int]],
    budget: Mapping[str, int],
    budget_scale: float,
    batch_size: int,
    workers: int,
    num_steps: int,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    validate_every: int,
    mask_ratio: float,
    aligned: Sequence[str],
    pooling: str,
    temperature: float,
    measure: str,
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
        sizes: How far a patch of each sensor runs along each axis it is cut on.
        budget: How many patches of each instrument a tile keeps.
        budget_scale: The most a training batch's budget is multiplied by.
        batch_size: How many tiles one batch holds.
        workers: How many processes read tiles beside the training.
        num_steps: How many steps the run takes.
        learning_rate: The peak learning rate, reached after the warmup.
        weight_decay: The AdamW weight decay.
        warmup_steps: How many steps the rate climbs before the cosine decay.
        validate_every: How many steps between two validations.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        aligned: The instruments MIM aligns, none for no MIM.
        pooling: What MIM aligns of each instrument, its GAP or its CLS.
        temperature: What MIM divides every cosine by.
        measure: The reconstruction error, MSE or RMSE.
        checkpoints: Where checkpoints are written, relative to the repository.
        seed: What fixes the masks, the order and the validation cuts.
        device: Where the model runs.
        run: The tracked run every metric is logged to.
        config: The run's whole config, saved beside the weights.

    Returns:
        last: The checkpoint the run ended on.
    """
    training = tile_loader(
        build, training, sizes, budget, budget_scale, seed, batch_size, workers, True
    )
    validation = tile_loader(
        build, validation, sizes, budget, budget_scale, seed, batch_size, workers, False
    )
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
    torch.set_float32_matmul_precision("high")

    def lambda_lr_schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(num_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lambda_lr_schedule)
    generators = mask_generators(model.encoders, seed, device)
    started = time.perf_counter()
    batches = chain.from_iterable(repeat(training))
    model.train()
    for step in range(1, num_steps + 1):
        batch, identities = next(batches)
        optimizer.zero_grad()
        terms = step_terms(
            model,
            batch,
            mask_ratio,
            aligned,
            pooling,
            temperature,
            measure,
            generators,
            device,
        )
        terms["loss"].backward()
        # Stabilize training against exploding gradients
        norm = clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError(
                f"step {step} has a non-finite gradient: {identities}"
            )
        optimizer.step()
        scheduler.step()
        log_terms(run, step, TRAIN, terms | {"grad_norm": norm})
        # The last step is validated too, so every run ends on a measured loss
        if step % validate_every and step < num_steps:
            continue
        metrics = validation_terms(
            model,
            validation,
            mask_ratio,
            aligned,
            pooling,
            temperature,
            measure,
            seed,
            device,
        )
        model.train()  # Validating switched it to evaluation
        log_terms(run, step, VALIDATION, metrics)
        log.info("step %d validation loss %.4f", step, metrics["loss"])
    last = checkpoint_path(checkpoints, "last")
    last.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": num_steps,
            "config": dict(config),
        },
        last,
    )
    run.summary.update(
        {"steps": num_steps, "run_seconds": time.perf_counter() - started}
    )
    return last
