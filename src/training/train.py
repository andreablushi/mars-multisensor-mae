"""Training the model over one split, validating over another, and keeping the best."""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from wandb.sdk.wandb_run import Run

from architecture.mae import CrossSensorMAE
from config.paths import REPO_ROOT
from logs.console import rich_logger
from logs.tracker import log_epoch, log_step, log_summary
from training.checkpoint import save_checkpoint
from training.early_stopping import EarlyStopping
from training.loss import csmae_loss
from training.masking import masked_reconstruction
from training.validate import validation_terms

BEST_CHECKPOINT = "best.pt"

log = rich_logger(__name__)


def train(
    model: CrossSensorMAE,
    training: DataLoader,
    validation: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    warmup_epochs: int,
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
        epochs: How many passes over the training tiles, at most.
        learning_rate: The peak learning rate, reached after the warmup.
        weight_decay: The AdamW weight decay.
        warmup_epochs: How many epochs the rate climbs before the cosine decay.
        patience: How many epochs without a lower validation loss before it stops.
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
    # Compute total steps and warmup duration
    steps = len(training) * epochs
    warmup = len(training) * warmup_epochs

    # Schedule function: linear warmup followed by cosine decay
    def lambda_lr_schedule(step: int) -> float:
        if step < warmup:
            return step / max(warmup, 1)  # Warmup phase
        progress = (step - warmup) / max(steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))  # Cosine decay phase

    scheduler = LambdaLR(optimizer, lambda_lr_schedule)
    stopping = EarlyStopping(patience)
    # Deterministic seed generator
    generator = torch.Generator(device=device).manual_seed(seed)
    best = REPO_ROOT / checkpoints / BEST_CHECKPOINT
    started = time.perf_counter()
    step, best_epoch = 0, -1
    for epoch in range(epochs):
        model.train()  # Enable training mode
        for batch, cells, _ in training:
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
        # Run validation pass after each epoch
        metrics = validation_terms(model, validation, mask_ratio, seed, device)
        log_epoch(run, step, epoch, metrics)
        log.info("epoch %d validation loss %.4f", epoch, metrics["loss"])
        # Track early stopping and persist best model weights
        if stopping.improved(metrics["loss"]):
            save_checkpoint(best, model, optimizer, epoch)
            best_epoch = epoch
        # Check early stopping patience trigger
        if stopping.stopped:
            log.info("no lower validation loss for %d epochs, stopping", patience)
            break
    # Record final run summary metadata
    log_summary(
        run,
        {
            "best_validation_loss": stopping.best,
            "best_epoch": best_epoch,
            "epochs_trained": epoch + 1,
            "steps": step,
            "run_seconds": time.perf_counter() - started,
        },
    )
    return best
