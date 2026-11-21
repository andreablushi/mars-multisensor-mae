"""Some instruments' tokens side by side, as a decoder reads them, and how they pool."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import NamedTuple

import torch
from torch import Tensor


class Pooling(StrEnum):
    """What one tile vector is made of an instrument's tokens."""

    CLS = "CLS"
    GAP = "GAP"
    GAP_SURFACE = "GAP-surface"


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

    @classmethod
    def joined(cls, parts: Sequence[Context]) -> Context:
        """Return some contexts side by side, in their order.

        Args:
            parts: The contexts.

        Returns:
            context: Their tokens, centres and which of them count, joined.
        """
        return cls(*(torch.cat(held, 1) for held in zip(*parts, strict=True)))
