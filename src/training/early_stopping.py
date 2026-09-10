"""Stopping a run once the validation loss has stopped falling."""

from __future__ import annotations

import math
from dataclasses import dataclass


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
