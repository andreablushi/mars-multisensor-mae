"""Keeping what a run has learnt, and picking it up again."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    path: Path, model: nn.Module, optimizer: Optimizer, epoch: int
) -> None:
    """Write the model and the optimizer down as they stand.

    Args:
        path: Where to write them, whose directory is made if missing.
        model: The model.
        optimizer: Its optimizer.
        epoch: How many epochs it has trained for.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        },
        path,
    )


def load_checkpoint(
    path: Path, model: nn.Module, optimizer: Optimizer | None = None
) -> int:
    """Return how many epochs a checkpoint trained for, after loading it.

    Args:
        path: Where it was written.
        model: The model to load it into, built the same way.
        optimizer: The optimizer to load it into, or None to load the model alone.

    Returns:
        epoch: How many epochs it had trained for.
    """
    held = torch.load(path, map_location="cpu")
    model.load_state_dict(held["model"])
    if optimizer is not None:
        optimizer.load_state_dict(held["optimizer"])
    return held["epoch"]
