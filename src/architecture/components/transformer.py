"""Attending over a set of tokens, skipping the ones that carry nothing."""

from __future__ import annotations

from torch import Tensor, nn

# B = batch, N = tokens, D = token channels.


class Transformer(nn.Module):
    """Pre-norm transformer blocks stacked, normalised once more at the end."""

    def __init__(self, dim: int, heads: int, depth: int) -> None:
        """Build the stack for one token width.

        Args:
            dim: The token width.
            heads: How many attention heads each block runs.
            depth: How many blocks are stacked.
        """
        super().__init__()
        # One pre-norm block: self-attention then a feed-forward four times as wide
        block = nn.TransformerEncoderLayer(
            dim,
            heads,
            4 * dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Stack the blocks and normalise once more at the end
        self.blocks = nn.TransformerEncoder(
            block, depth, norm=nn.LayerNorm(dim), enable_nested_tensor=False
        )

    def forward(self, tokens: Tensor, attended: Tensor) -> Tensor:
        """Return the tokens after attending over the ones that carry something.

        Args:
            tokens: The tokens to attend over. (B, N, D)
            attended: Which of them carry something. (B, N)

        Returns:
            tokens: The attended tokens. (B, N, D)
        """
        # Invert valid mask: PyTorch key_padding_mask expects True for tokens to ignore
        padding = ~attended  # (B, N)
        # Unmask wholly empty sequences to keep the soft-max from going nan
        padding[padding.all(dim=1)] = False
        # Every token attends over the ones that carry something
        return self.blocks(tokens, src_key_padding_mask=padding)  # (B, N, D)
