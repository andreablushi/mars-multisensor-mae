"""Training the model over one split for a fixed schedule, checking it on another."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import chain, repeat
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
from training.step import step_terms
from training.validate import validation_terms

log = console_logger(__name__)

CROSSENCODER = "crossencoder."


def train(
    model: CrossSensorMAE,
    training: DataLoader,
    validation: DataLoader,
    epochs: int,
    learning_rate: float,
    crossencoder_rate: float,
    weight_decay: float,
    warmup_epochs: int,
    validate_every: int,
    mask_ratio: float,
    sources: Mapping[str, Sequence[str]],
    frozen: Sequence[str],
    checkpoints: str,
    seed: int,
    device: torch.device,
    run: Run,
) -> Path:
    """Return where the last checkpoint was written, after training the model.

    Args:
        model: The model, already on the device.
        training: The training split, in batches.
        validation: The validation split, in batches.
        epochs: How many passes over the training split the run takes.
        learning_rate: The peak learning rate, reached after the warmup.
        crossencoder_rate: What the crossencoder's learning rate is scaled by.
        weight_decay: The AdamW weight decay.
        warmup_epochs: How many epochs the rate climbs before the cosine decay.
        validate_every: How many epochs between two validations.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        sources: The instruments each instrument is rebuilt from cross-modally.
        frozen: The modules left as they are, named as the model holds them.
        checkpoints: Where checkpoints are written, relative to the repository.
        seed: What fixes the masks.
        device: Where the model runs.
        run: The tracked run every metric is logged to.

    Returns:
        last: The checkpoint the run ended on.
    """
    for name in frozen:
        model.get_submodule(name).requires_grad_(False)
    groups = defaultdict(list)
    for name, one in model.named_parameters():
        if one.requires_grad:
            groups[name.startswith(CROSSENCODER), one.ndim > 1].append(one)
    # Biases, norms and learned vectors are not decayed, as MAE leaves them
    optimizer = torch.optim.AdamW(
        [
            {
                "params": held,
                "lr": learning_rate * (crossencoder_rate if shared else 1.0),
                "weight_decay": weight_decay if decayed else 0.0,
            }
            for (shared, decayed), held in groups.items()
        ],
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        fused=True,
    )
    torch.set_float32_matmul_precision("high")
    max_steps = epochs * len(training)
    warmup_steps = warmup_epochs * len(training)

    def lambda_lr_schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lambda_lr_schedule)
    generator = torch.Generator(device=device).manual_seed(seed)
    started = time.perf_counter()
    batches = chain.from_iterable(repeat(training))
    model.train()
    for step in range(1, max_steps + 1):
        began = time.perf_counter()
        batch, identities = next(batches)
        waited = time.perf_counter() - began
        optimizer.zero_grad()
        terms = step_terms(model, batch, mask_ratio, sources, generator, device)
        terms["loss"].backward()
        # Stabilize training against exploding gradients
        norm = clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError(
                f"step {step} has a non-finite gradient: {identities}"
            )
        optimizer.step()
        scheduler.step()
        log_step(run, step, terms | {"grad_norm": norm})
        log.info(
            "step %d waited %.1f s for data, computed in %.1f s",
            step,
            waited,
            time.perf_counter() - began - waited,
        )
        # The last step is validated too, so every run ends on a measured loss
        if step % (validate_every * len(training)) and step < max_steps:
            continue
        metrics = validation_terms(model, validation, mask_ratio, sources, seed, device)
        model.train()  # Validating switched it to evaluation
        log_validation(run, step, metrics)
        log.info("step %d validation loss %.4f", step, metrics["loss"])
    last = checkpoint_path(checkpoints, "last")
    last.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": max_steps,
        },
        last,
    )
    log_summary(
        run,
        {
            "validation_loss": metrics["loss"],
            "steps": max_steps,
            "run_seconds": time.perf_counter() - started,
        },
    )
    return last
