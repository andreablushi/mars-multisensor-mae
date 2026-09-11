"""The cross-sensor encoder, one set of weights every instrument passes through."""

from __future__ import annotations

from torch import Tensor, nn

from architecture.components.transformer import Transformer


class CrossSensorEncoder(nn.Module):
    """A transformer applied to each instrument's tokens on their own, weights shared.

    Attributes:
        blocks: The transformer.
    """

    def __init__(self, dim: int, heads: int, depth: int) -> None:
        """Build the shared stack.

        Args:
            dim: The token width.
            heads: How many attention heads each block runs.
            depth: How many blocks are stacked.
        """
        super().__init__()
        self.blocks = Transformer(dim, heads, depth)

    def forward(self, tokens: Tensor, visible: Tensor) -> Tensor:
        """Return one instrument's tokens mapped into the space every instrument shares.

        Args:
            tokens: The instrument's encoded tokens. (B, K, D)
            visible: Which of them carry a patch the encoder read. (B, K)

        Returns:
            tokens: The mapped tokens, meaningful where visible. (B, K, D)
        """
        return self.blocks(tokens, visible)  # (B, K, D)
