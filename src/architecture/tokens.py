"""What one stage hands the next: a sensor's patches, and the tokens a decoder reads."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

# B = batch, K = patches, S = tokens read together, P = patch dimensions,
# D = token channels.


class Tokens(NamedTuple):
    """One sensor's patches over a batch of tiles, padded to one count.

    Attributes:
        values: The normalised patches, zero where padded. (B, K, *P)
        measured: Whether each sample is a measurement, false over padding. (B, K, *P')
        condition: What each patch is conditioned on, flattened, or nothing. (B, K, C)
        position: The patch centre, east and north in metres, delay in rows. (B, K, 3)
    """

    values: Tensor
    measured: Tensor
    condition: Tensor
    position: Tensor

    @property
    def present(self) -> Tensor:
        """Return which slots hold a patch rather than padding.

        Returns:
            present: True where any sample of the slot is a measurement. (B, K)
        """
        return self.measured.flatten(2).any(dim=-1)

    def led_by_cls(self, readable: Tensor) -> tuple[Tensor, Tensor]:
        """Return which slots count and where they sit, a CLS slot leading them.

        Args:
            readable: Which patches count. (B, K)

        Returns:
            readable: The same, the CLS slot always counted. (B, 1 + K)
            position: The patch centres, the CLS slot at the tile centre, which the
                rotary encoding leaves unturned. (B, 1 + K, 3)
        """
        return (
            torch.cat([readable.new_ones(len(readable), 1), readable], 1),
            torch.cat([self.position.new_zeros(len(readable), 1, 3), self.position], 1),
        )

    def kept(self, keep: Tensor) -> tuple[Tokens, Tensor]:
        """Return the kept slots alone, packed to the front and padded to one count.

        Args:
            keep: Which slots are kept. (B, K)

        Returns:
            tokens: The kept patches, in their order. (B, K', *P)
            kept: Which of the packed slots hold a kept patch. (B, K')
        """
        order = keep.int().argsort(dim=1, descending=True, stable=True)
        order = order[:, : int(keep.sum(1).max())]  # (B, K')

        def taken(one: Tensor) -> Tensor:
            shape = (*order.shape, *[1] * (one.dim() - 2))
            return one.gather(1, order.view(shape).expand(-1, -1, *one.shape[2:]))

        return Tokens(*(taken(one) for one in self)), keep.gather(1, order)


class Context(NamedTuple):
    """Some instruments' tokens side by side, as a decoder reads them.

    Attributes:
        tokens: The tokens, in the space every instrument shares. (B, S, D)
        position: Each one's patch centre, in metres or rows. (B, S, 3)
        readable: Which of them count. (B, S)
    """

    tokens: Tensor
    position: Tensor
    readable: Tensor


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
        # Every array a model reads padded with zeros to the most patches any tile has
        padded = {
            key: pad_sequence(
                [torch.as_tensor(one[key]) for one in held], batch_first=True
            )
            for key in Tokens._fields
        }
        # Values in float32, whatever the stored type
        padded["values"] = padded["values"].float()
        # The instrument's patches over the batch
        batch[name] = Tokens(**padded)
    # The patches and the tile names, in the batch's order
    return batch, [identity for _, identity in samples]
