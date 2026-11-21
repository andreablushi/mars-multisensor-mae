"""Some instruments' tokens side by side, as a decoder reads them, and how they pool."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import NamedTuple

import torch
from torch import Tensor
from torch.nn import functional


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


class EncodedBatch(NamedTuple):
    """Every instrument's tokens in the shared space, and the token read over them all.

    Attributes:
        cls: The token read over every instrument together. (B, D)
        instruments: Each instrument's CLS token, then its tokens, with their centres
            and which of them count. (B, 1 + K, D)
    """

    cls: Tensor
    instruments: dict[str, Context]

    def unimodal_reconstruction_context(self, name: str) -> Context:
        """Return what UMR rebuilds one instrument from: the overall CLS, then its own.

        Args:
            name: The instrument rebuilt.

        Returns:
            context: The overall CLS at the tile centre, then the instrument's own
                context.
        """
        lead = Context(
            self.cls.unsqueeze(1),
            self.cls.new_zeros(len(self.cls), 1, 3),
            self.cls.new_ones(len(self.cls), 1, dtype=torch.bool),
        )
        return Context.joined([lead, self.instruments[name]])

    def cross_modal_reconstruction_context(self, name: str) -> Context:
        """Return what CMR rebuilds one instrument from: every other's, no overall CLS.

        Args:
            name: The instrument rebuilt.

        Returns:
            context: Every other instrument's context, in their order.
        """
        return Context.joined(
            [held for other, held in self.instruments.items() if other != name]
        )

    def pooled_tile_vectors(
        self,
        pooling: Pooling,
        readable: Mapping[str, Tensor] | None = None,
        unit: bool = True,
    ) -> dict[str, Tensor]:
        """Return each instrument's tile vector, the CLS or GAP of its readable tokens.

        Args:
            pooling: What a tile vector is, "CLS" or "GAP".
            readable: Which of each instrument's tokens GAP averages, by default the
                ones its context counts. (B, K)
            unit: Whether each vector is normalised, else kept raw, GAP their mean.

        Returns:
            pooled: Each instrument's tile vector, in float. (B, D)
        """
        pooled = {}
        for name, held in self.instruments.items():
            if pooling == Pooling.CLS:
                vector = held.tokens[:, 0].float()  # (B, D)
            else:
                counted = held.readable[:, 1:] if readable is None else readable[name]
                vector = (held.tokens[:, 1:].float() * counted.unsqueeze(-1)).sum(1)
                if not unit:
                    vector = vector / counted.sum(1, keepdim=True).clamp(min=1)
            pooled[name] = functional.normalize(vector, dim=-1) if unit else vector
        return pooled
