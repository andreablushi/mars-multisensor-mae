"""Training the model over one split, validating over another, and keeping the best."""

from __future__ import annotations

import math
import time
from itertools import islice
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from wandb.sdk.wandb_run import Run

from architecture.mae import CrossSensorMAE
from config.paths import REPO_ROOT
from logs.console import rich_logger
from logs.tracker import log_step, log_summary, log_validation
from training.checkpoint import save_checkpoint
from training.early_stopping import EarlyStopping
from training.loss import csmae_loss
from training.masking import masked_reconstruction
from training.validate import validation_terms

BEST_CHECKPOINT = "best.pt"

log = rich_logger(__name__)


def endless(loader: DataLoader):
    """Yield the loader's batches over and over, so a run is counted in steps.

    Args:
        loader: The split to read, in batches.

    Yields:
        batch: What one step reads, the loader started again once it runs out.
    """
    while True:
        yield from loader


def train(
    model: CrossSensorMAE,
    training: DataLoader,
    validation: DataLoader,
    max_steps: int,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    validate_every: int,
    patience: int,
    mask_ratio: float,
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
        learning_rate: The peak learning rate, reached after the warmup.
        weight_decay: The AdamW weight decay.
        warmup_steps: How many steps the rate climbs before the cosine decay.
        validate_every: How many steps between two validations.
        patience: How many validations without a lower loss before the run stops.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        checkpoints: Where checkpoints are written, relative to the repository.
        seed: What fixes the masks.
        device: Where the model runs.
        run: The tracked run every metric is logged to.

    Returns:
        best: The checkpoint with the lowest validation loss.
    """
    # Configure AdamW optimizer with custom hyperparameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )

    # Schedule function: linear warmup followed by cosine decay
    def lambda_lr_schedule(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)  # Warmup phase
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))  # Cosine decay phase

    scheduler = LambdaLR(optimizer, lambda_lr_schedule)
    stopping = EarlyStopping(patience)
    # Deterministic seed generator
    generator = torch.Generator(device=device).manual_seed(seed)
    best = REPO_ROOT / checkpoints / BEST_CHECKPOINT
    started = time.perf_counter()
    step, best_step = 0, -1
    model.train()  # Enable training mode
    for batch, cells, _ in islice(endless(training), max_steps):
        batch, reconstruction = masked_reconstruction(
            model, batch, cells, mask_ratio, generator, device
        )
        # Compute cross-sensor MAE loss terms
        terms = csmae_loss(reconstruction, batch)
        # Backpropagation pass
        optimizer.zero_grad()
        terms["loss"].backward()
        # Stabilize training against exploding gradients
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1
        log_step(run, step, terms)  # Log step-level metrics to experiment tracker
        if step % validate_every:
            continue
        metrics = validation_terms(model, validation, mask_ratio, seed, device)
        model.train()  # Validating switched it to evaluation
        log_validation(run, step, metrics)
        log.info("step %d validation loss %.4f", step, metrics["loss"])
        # Track early stopping and persist best model weights
        if stopping.improved(metrics["loss"]):
            save_checkpoint(best, model, optimizer, step)
            best_step = step
        # Check early stopping patience trigger
        if stopping.stopped:
            log.info("no lower validation loss for %d validations, stopping", patience)
            break
    # Record final run summary metadata
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
