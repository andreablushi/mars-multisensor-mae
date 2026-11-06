"""Telling whether a tile is icy at each depth from one vector, a linear probe."""

from __future__ import annotations

from torch import Tensor, nn

from architecture.models.context import Context


class Classifier(nn.Module):
    """A linear layer from a tile's GAP or CLS to one logit per depth.

    Attributes:
        pooling: What it reads of a tile, "GAP" or "CLS".
        linear: From that vector to the logits.
    """

    def __init__(self, dim: int, depths: int, pooling: str) -> None:
        """Build the classifier for one token width.

        Args:
            dim: The token width.
            depths: How many depths a tile is labelled at.
            pooling: What it reads of a tile, "GAP" or "CLS".
        """
        super().__init__()
        self.pooling = pooling
        self.linear = nn.Linear(dim, depths)

    def forward(self, cls: Tensor, context: Context) -> Tensor:
        """Return each tile's logits.

        Args:
            cls: The token read over every instrument together. (B, D)
            context: Every instrument's tokens side by side, without their CLS.

        Returns:
            logits: One per depth. (B, Z)
        """
        if self.pooling == "CLS":
            return self.linear(cls)
        readable = context.readable.unsqueeze(-1)  # (B, S, 1)
        return self.linear((context.tokens.float() * readable).sum(1) / readable.sum(1))
