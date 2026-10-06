"""Bringing the sensors' tokens into one space, attended over together."""

from __future__ import annotations

from torch import Tensor, nn

from architecture.components.transformer import Transformer

# B = batch, K = patches, D = token channels.


class CrossSensorEncoder(nn.Module):
    """Attend over one sensor's tokens at a time, on weights every sensor shares.

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
        # The stack the instruments' tokens pass through together
        self.blocks = Transformer(dim, heads, depth)

    def forward(self, keys: Tensor, visible: Tensor, position: Tensor) -> Tensor:
        """Return the tokens mapped into the space every instrument shares.

        Args:
            keys: One instrument's encoded tokens, each a key and a query. (B, K, D)
            visible: Which of them carry a patch the encoder read. (B, K)
            position: Each token's patch centre, in metres or rows. (B, K, 3)

        Returns:
            tokens: The mapped tokens, meaningful where visible. (B, K, D)
        """
        # Project instrument tokens into the shared space, masking unread patches
        return self.blocks(keys, visible, position)  # (B, K, D)
