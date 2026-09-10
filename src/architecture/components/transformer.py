"""A stack of transformer blocks, read with a mask over the tokens to skip."""

from __future__ import annotations

from torch import Tensor, nn


class Transformer(nn.Module):
    """Pre-norm transformer blocks stacked, normalised once more at the end.

    Attributes:
        blocks: The stack.
    """

    def __init__(self, dim: int, heads: int, depth: int) -> None:
        """Build the stack for one token width.

        Args:
            dim: The token width.
            heads: How many attention heads each block runs.
            depth: How many blocks are stacked.
        """
        super().__init__()
        block = nn.TransformerEncoderLayer(
            dim,
            heads,
            4 * dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            block, depth, norm=nn.LayerNorm(dim), enable_nested_tensor=False
        )

    def forward(self, tokens: Tensor, attended: Tensor) -> Tensor:
        """Return the tokens after attending over the ones that carry something.

        Args:
            tokens: One per slot. (B, N, D)
            attended: Which of them carry something to attend to. (B, N)

        Returns:
            tokens: One per slot, meaningful where attended. A row with nothing
                to attend to skips no key, so it comes out finite and is
                dropped downstream. (B, N, D)
        """
        padding = ~attended  # (B, N)
        padding[padding.all(dim=1)] = False
        return self.blocks(tokens, src_key_padding_mask=padding)  # (B, N, D)
