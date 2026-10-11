"""Keeping what a run has learnt, picking it up again, and when to stop."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    path: Path, model: nn.Module, optimizer: Optimizer, step: int
) -> None:
    """Write the model and the optimizer down as they stand.

    Args:
        path: Where to write them, whose directory is made if missing.
        model: The model.
        optimizer: Its optimizer.
        step: How many steps it has trained for.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module) -> int:
    """Return how many steps a checkpoint trained for, after loading it.

    Args:
        path: Where it was written.
        model: The model to load it into, built the same way.

    Returns:
        step: How many steps it had trained for.
    """
    held = torch.load(path, map_location="cpu")
    model.load_state_dict(held["model"])
    return held["step"]


@dataclass
class EarlyStopping:
    """How long a run is given to beat its best validation loss.

    Attributes:
        patience: How many validations in a row may fail to beat it.
        best: The lowest validation loss so far.
        waited: How many validations in a row have failed to beat it.
    """

    patience: int
    best: float = math.inf
    waited: int = 0

    def improved(self, loss: float) -> bool:
        """Return whether one more validation beat the best, else count the wait.

        Args:
            loss: The validation loss just measured.

        Returns:
            improved: Whether it is the lowest so far.
        """
        if loss < self.best:
            self.best = loss
            self.waited = 0
            return True
        self.waited += 1
        return False

    @property
    def stopped(self) -> bool:
        """Return whether the run has waited as long as it is given.

        Returns:
            stopped: Whether the patience is spent.
        """
        return self.waited >= self.patience
