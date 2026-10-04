"""Bringing every sensor's tokens into one space, on weights they all pass through."""

from __future__ import annotations

from torch import Tensor, nn

from architecture.components.transformer import Transformer

# B = batch, K = patches, D = token channels.


class CrossSensorEncoder(nn.Module):
    """Attend over one sensor's tokens alone, on weights shared with every other.

    Attributes:
        blocks: The transformer.
    """

    def __init__(self, dim: int, heads: int, depth: int, radius: float) -> None:
        """Build the shared stack.

        Args:
            dim: The token width.
            heads: How many attention heads each block runs.
            depth: How many blocks are stacked.
            radius: How far apart on the ground two tokens may attend, in metres.
        """
        super().__init__()
        # The stack every instrument's tokens pass through, on the same weights
        self.blocks = Transformer(dim, heads, depth, radius)

    def forward(self, tokens: Tensor, visible: Tensor, position: Tensor) -> Tensor:
        """Return one instrument's tokens mapped into the space every instrument shares.

        Args:
            tokens: The instrument's encoded tokens. (B, K, D)
            visible: Which of them carry a patch the encoder read. (B, K)
            position: Where each token's patch sits and reaches, in metres. (B, K, 6)

        Returns:
            tokens: The mapped tokens, meaningful where visible. (B, K, D)
        """
        # Project instrument tokens into the shared space, masking unread patches
        return self.blocks(tokens, visible, position)  # (B, K, D)
