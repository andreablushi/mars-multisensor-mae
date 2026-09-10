"""What the model is handed: one instrument's patches over a batch of features."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class Tokens:
    """One instrument's patches over a batch of features, padded to one count.

    Attributes:
        values: The normalised patches, zero where padded. (B, K, *P)
        valid: Whether each sample of a patch is a measurement, over the ground
            axes and broadcastable to the values. (B, K, *P')
        position: How far east and north of the feature centre each patch
            centre sits, and how high above the areoid, in metres. (B, K, 3)
        visible: Whether each slot holds a patch its encoder may read, which a
            hidden patch and padding do not. (B, K)
        present: Whether each slot holds a patch rather than padding. (B, K)
    """

    values: Tensor
    valid: Tensor
    position: Tensor
    visible: Tensor
    present: Tensor

    def to(self, device: torch.device) -> Tokens:
        """Return the same tokens held on one device.

        Args:
            device: Where the model runs.

        Returns:
            tokens: The tokens, every array moved there.
        """
        return Tokens(
            self.values.to(device),
            self.valid.to(device),
            self.position.to(device),
            self.visible.to(device),
            self.present.to(device),
        )
