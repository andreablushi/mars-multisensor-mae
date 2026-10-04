"""Training the model over one split, validating over another, and keeping the best."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from itertools import chain, islice, repeat
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from wandb.sdk.wandb_run import Run

from architecture.mae import CrossSensorMAE
from configs.paths import checkpoint_path
from logs.console import console_logger
from logs.tracker import log_step, log_summary, log_validation
from training.checkpoint import save_checkpoint
from training.early_stopping import EarlyStopping
from training.loss import pooled_terms
from training.step import device_batch, drawn_masks, masked_sums
from training.validate import validation_terms

log = console_logger(__name__)


def train(
    model: CrossSensorMAE,
    training: DataLoader,
    validation: DataLoader,
    max_steps: int,
    accumulate: int,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    validate_every: int,
    patience: int,
    mask_ratio: float,
    drop_ratio: float,
    checkpoints: str,
    seed: int,
    device: torch.device,
    run: Run,
) -> Path:
    """Return where the best checkpoint was written, after training the model.

    Args:
        model: The model, already on the device.
        training: The training split, in batches.
        validation: The validation split, in batches.
        max_steps: How many steps the run takes, at most.
        accumulate: How many batches one step adds its gradients over, pooled
            so the step equals one batch of all their tiles.
        learning_rate: The peak learning rate, reached after the warmup.
        weight_decay: The AdamW weight decay.
        warmup_steps: How many steps the rate climbs before the cosine decay.
        validate_every: How many steps between two validations.
        patience: How many validations without a lower loss before the run stops.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        drop_ratio: The chance each instrument is left out of the grid.
        checkpoints: Where checkpoints are written, relative to the repository.
        seed: What fixes the masks.
        device: Where the model runs.
        run: The tracked run every metric is logged to.

    Returns:
        best: The checkpoint with the lowest validation loss.
    """
    # Biases, norms and learned vectors are not decayed, as MAE leaves them
    decayed = [one for one in model.parameters() if one.ndim > 1]
    kept = [one for one in model.parameters() if one.ndim <= 1]
    optimizer = torch.optim.AdamW(
        [{"params": decayed}, {"params": kept, "weight_decay": 0.0}],
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )

    def lambda_lr_schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lambda_lr_schedule)
    stopping = EarlyStopping(patience)
    generator = torch.Generator(device=device).manual_seed(seed)
    best = checkpoint_path(checkpoints, "best")
    started = time.perf_counter()
    batches = chain.from_iterable(repeat(training))
    best_step = -1
    model.train()
    for step in range(1, max_steps + 1):
        began = time.perf_counter()
        passes = list(islice(batches, accumulate))
        waited = time.perf_counter() - began
        optimizer.zero_grad()
        drawn, counts = [], defaultdict(float)
        for batch, cells, _ in passes:
            batch, _ = device_batch(batch, cells, device)
            visible, hidden, kept, held = drawn_masks(
                batch, mask_ratio, drop_ratio, generator
            )
            drawn.append((visible, hidden, kept))
            for term, count in held.items():
                counts[term] += count
        sums = defaultdict(float)
        for (batch, cells, _), (visible, hidden, kept) in zip(
            passes, drawn, strict=True
        ):
            batch, cells = device_batch(batch, cells, device)
            held = masked_sums(model, batch, cells, visible, hidden, kept)
            pooled_terms(held, counts)["loss"].backward()
            for term, value in held.items():
                sums[term] += value.detach()
        terms = pooled_terms(sums, counts)
        # Stabilize training against exploding gradients
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        log_step(run, step, terms)
        log.info(
            "step %d waited %.1f s for data, computed in %.1f s",
            step,
            waited,
            time.perf_counter() - began - waited,
        )
        # The last step is validated too, so a short run still leaves a checkpoint
        if step % validate_every and step < max_steps:
            continue
        metrics = validation_terms(
            model, validation, mask_ratio, drop_ratio, seed, device
        )
        model.train()  # Validating switched it to evaluation
        log_validation(run, step, metrics)
        log.info("step %d validation loss %.4f", step, metrics["loss"])
        if stopping.improved(metrics["loss"]):
            save_checkpoint(best, model, optimizer, step)
            best_step = step
        if stopping.stopped:
            log.info("no lower validation loss for %d validations, stopping", patience)
            break
    log_summary(
        run,
        {
            "best_validation_loss": stopping.best,
            "best_step": best_step,
            "steps": step,
            "run_seconds": time.perf_counter() - started,
        },
    )
    return best
