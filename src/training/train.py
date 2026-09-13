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
from config.schema import Config
from logs.console import logger
from logs.tracker import log_epoch, log_step, log_summary
from training.checkpoint import save_checkpoint
from training.early_stopping import EarlyStopping
from training.loss import csmae_loss
from training.masking import random_correspondence
from training.validate import validate

BEST_CHECKPOINT = "best.pt"

log = logger(__name__)


def train(
    model: CrossSensorMAE,
    training: DataLoader,
    validation: DataLoader,
    config: Config,
    device: torch.device,
    run: Run,
) -> Path:
    """Return where the best checkpoint was written, after training the model.

    Args:
        model: The model, already on the device.
        training: The training split, in batches.
        validation: The validation split, in batches.
        config: How to train, validate and stop.
        device: Where the model runs.
        run: The tracked run every metric is logged to.

    Returns:
        best: The checkpoint with the lowest validation loss.
    """
    settings = config.training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
        betas=(0.9, 0.95),
    )
    steps = len(training) * settings.epochs
    warmup = len(training) * settings.warmup_epochs

    def schedule(step: int) -> float:
        """Return the share of the peak learning rate one step runs at."""
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, schedule)
    stopping = EarlyStopping(settings.patience)
    generator = torch.Generator(device=device).manual_seed(config.dataset.seed)
    best = REPO_ROOT / settings.checkpoints / BEST_CHECKPOINT
    started = time.perf_counter()
    step, best_epoch = 0, -1
    for epoch in range(settings.epochs):
        model.train()
        for batch, _ in training:
            batch = {name: tokens.to(device) for name, tokens in batch.items()}
            batch = random_correspondence(batch, settings.mask_ratio, generator)
            terms = csmae_loss(model(batch), batch, settings.temperature)
            rate = scheduler.get_last_lr()[0]
            optimizer.zero_grad()
            terms["loss"].backward()
            # One patch a sounder wrote badly must not carry the whole run off.
            gradient = clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            log_step(
                run,
                step,
                terms,
                {
                    "epoch": epoch,
                    "learning_rate": rate,
                    "gradient_norm": float(gradient),
                },
            )
        metrics = validate(model, validation, config, device)
        log_epoch(run, step, epoch, metrics)
        log.info("epoch %d validation loss %.4f", epoch, metrics["loss"])
        if stopping.improved(metrics["loss"]):
            save_checkpoint(best, model, optimizer, epoch)
            best_epoch = epoch
        if stopping.stopped:
            log.info(
                "no lower validation loss for %d epochs, stopping", settings.patience
            )
            break
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
