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
            self.channels.to(device),
            self.visible.to(device),
            self.present.to(device),
        )


def collate(
    samples: list[tuple[dict[str, dict[str, np.ndarray]], tuple[str, str]]],
) -> tuple[dict[str, Tokens], list[tuple[str, str]]]:
    """Return one batch of every instrument's tokens, and whose each read is.

    Args:
        samples: What the dataset read of each feature of the batch.

    Returns:
        batch: Each sensor's patches padded to the most any read holds, none hidden.
        identities: The feature each read belongs to, in the batch's order.
    """
    batch = {}
    for name in samples[0][0]:
        held = [sample[name] for sample, _ in samples]
        counts = torch.tensor([len(one["values"]) for one in held])  # (B,)
        slots = torch.arange(int(counts.max()))  # (K,)
        padded = {
            key: pad_sequence(
                [torch.as_tensor(one[key]) for one in held], batch_first=True
            )
            for key in held[0]
        }
        present = slots.unsqueeze(0) < counts.unsqueeze(1)  # (B, K)
        batch[name] = Tokens(**padded, visible=present, present=present)
    return batch, [identity for _, identity in samples]
