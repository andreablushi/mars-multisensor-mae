"""What the model is handed: one sensor's patches over a batch of features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


@dataclass(frozen=True, slots=True)
class Tokens:
    """One sensor's patches over a batch of features, padded to one count.

    Attributes:
        values: The normalised patches, zero where padded. (B, K, *P)
        valid: Whether each sample is a measurement. (B, K, *P')
        position: The patch centre and its span, in metres. (B, K, 6)
        channels: What each channel of each patch measures, in its own unit. (B, K, C)
        visible: Whether a slot holds a patch its encoder may read. (B, K)
        present: Whether each slot holds a patch rather than padding. (B, K)
    """

    values: Tensor
    valid: Tensor
    position: Tensor
    channels: Tensor
    visible: Tensor
    present: Tensor

    def to(self, device: torch.device) -> Tokens:
        """Return the same tokens held on one device."""
        # Transfer all underlying data and mask tensors to target execution device
        return Tokens(
            self.values.to(device),
            self.valid.to(device),
            self.position.to(device),
            self.channels.to(device),
            self.visible.to(device),
            self.present.to(device),
        )


def collate(
    samples: list[tuple[dict[str, dict[str, np.ndarray]], tuple[str, str]]],
) -> tuple[dict[str, Tokens], list[tuple[str, str]]]:
    """Return one batch of every instrument's tokens, and whose each read is."""
    batch = {}
    # Process each instrument/sensor present in the first dataset sample
    for name in samples[0][0]:
        held = [sample[name] for sample, _ in samples]
        # Count number of active patches per sample to find max sequence length K
        counts = torch.tensor([len(one["values"]) for one in held])  # (B,)
        slots = torch.arange(int(counts.max()))  # (K,)
        # Pad variable-length patch arrays across batch samples to uniform length K
        padded = {
            key: pad_sequence(
                [torch.as_tensor(one[key]) for one in held], batch_first=True
            )
            for key in held[0]
        }
        # Construct boolean mask identifying real patches versus zero-padded slots
        present = slots.unsqueeze(0) < counts.unsqueeze(1)  # (B, K)
        # Instantiate Tokens container (defaulting visible patches to present patches)
        batch[name] = Tokens(**padded, visible=present, present=present)
    # Return dictionary of sensor Tokens and associated feature identity tuples
    return batch, [identity for _, identity in samples]
