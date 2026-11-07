"""Telling how consistent with ice a tile is from one vector, by one linear layer."""

from __future__ import annotations

from torch import Tensor, nn

from architecture.models.context import Context


class Classifier(nn.Module):
    """A linear layer from a tile's GAP or CLS to its consistency at each depth.

    Attributes:
        pooling: What it reads of a tile, "GAP" or "CLS".
        linear: From that vector to the consistency, after a BatchNorm without affine
            when the encoders are frozen, as MAE's linear probe.
    """

    def __init__(self, dim: int, depths: int, pooling: str, probed: bool) -> None:
        """Build the classifier for one token width.

        Args:
            dim: The token width.
            depths: How many depths a tile is labelled at.
            pooling: What it reads of a tile, "GAP" or "CLS".
            probed: Whether the encoders stay frozen, so the vector is normalised.
        """
        super().__init__()
        self.pooling = pooling
        self.linear = nn.Sequential(
            *([nn.BatchNorm1d(dim, affine=False, eps=1e-6)] if probed else []),
            nn.Linear(dim, depths),
        )

    def forward(self, cls: Tensor, context: Context) -> Tensor:
        """Return each tile's consistency.

        Args:
            cls: The token read over every instrument together. (B, D)
            context: Every instrument's tokens side by side, without their CLS.

        Returns:
            consistency: One per depth. (B, Z)
        """
        if self.pooling == "CLS":
            return self.linear(cls)
        readable = context.readable.unsqueeze(-1)  # (B, S, 1)
        return self.linear((context.tokens.float() * readable).sum(1) / readable.sum(1))
