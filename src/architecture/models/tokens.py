"""One sensor's patches over a batch of tiles, as every stage reads them."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor
from torch.nn import functional


class Tokens(NamedTuple):
    """One sensor's patches over a batch of tiles, padded to one count.

    Attributes:
        values: The normalised patches, zero where padded. (B, K, *P)
        measured: Whether each sample is a measurement, false over padding. (B, K, *P')
        condition: What each patch is conditioned on, flattened, or nothing. (B, K, C)
        acquisition: The acquisition angles of each patch's observation, in degrees,
            or nothing. (B, K, A)
        position: The patch centre, east and north in metres, delay in rows. (B, K, 3)
    """

    values: Tensor
    measured: Tensor
    condition: Tensor
    acquisition: Tensor
    position: Tensor

    @property
    def present(self) -> Tensor:
        """Return which slots hold a patch rather than padding.

        Returns:
            present: True where any sample of the slot is a measurement. (B, K)
        """
        return self.measured.flatten(2).any(dim=-1)

    def to(self, device: torch.device) -> Tokens:
        """Return the same patches on one device.

        Args:
            device: Where they are moved.

        Returns:
            tokens: The patches, every array on the device.
        """
        return Tokens(*(one.to(device, non_blocking=True) for one in self))

    def readable_and_position_with_cls(self, readable: Tensor) -> tuple[Tensor, Tensor]:
        """Return which slots count and where they sit, a CLS slot leading them.

        Args:
            readable: Which patches count. (B, K)

        Returns:
            readable: The same, the CLS slot always counted. (B, 1 + K)
            position: The patch centres, the CLS slot at the tile centre, which the
                rotary encoding leaves unturned. (B, 1 + K, 3)
        """
        return (
            functional.pad(readable, (1, 0), value=True),
            functional.pad(self.position, (0, 0, 1, 0)),
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
        taken = (
            torch.take_along_dim(
                one, order.view(*order.shape, *[1] * (one.dim() - 2)), 1
            )
            for one in self
        )
        return Tokens(*taken), keep.gather(1, order)
