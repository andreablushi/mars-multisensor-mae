"""What one stage hands the next: a sensor's patches."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

# B = batch, K = patches, P = patch dimensions.


class Tokens(NamedTuple):
    """One sensor's patches over a batch of tiles, padded to one count.

    Attributes:
        values: The normalised patches, zero where padded. (B, K, *P)
        measured: Whether each sample is a measurement, false over padding. (B, K, *P')
        position: The patch centre and its span, in metres or rows. (B, K, 6)
    """

    values: Tensor
    measured: Tensor
    position: Tensor

    @property
    def present(self) -> Tensor:
        """Return which slots hold a patch rather than padding.

        Returns:
            present: True where any sample of the slot is a measurement. (B, K)
        """
        return self.measured.flatten(2).any(dim=-1)


def token_batch_padding(
    samples: list[tuple[dict[str, dict[str, np.ndarray]], str]],
) -> tuple[dict[str, Tokens], list[str]]:
    """Return one batch of every instrument's tokens, and whose.

    Args:
        samples: Each tile's patch arrays and its identity.

    Returns:
        batch: Each instrument's patches over the batch, keyed as ODE names it.
        identities: The tile each read belongs to, in the batch's own order.
    """
    batch = {}
    # Pad each instrument's patches across the batch's tiles
    for name in samples[0][0]:
        # This instrument's arrays in every tile
        held = [sample[name] for sample, _ in samples]
        # Every array padded with zeros to the most patches any tile has
        padded = {
            key: pad_sequence(
                [torch.as_tensor(one[key]) for one in held], batch_first=True
            )
            for key in held[0]
        }
        # Values in float32, whatever the stored type
        padded["values"] = padded["values"].float()
        # The instrument's patches over the batch
        batch[name] = Tokens(**padded)
    # The patches and the tile names, in the batch's order
    return batch, [identity for _, identity in samples]
