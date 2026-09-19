"""Attending over a set of tokens, skipping the ones that carry nothing."""

from __future__ import annotations

from torch import Tensor, nn


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
        # Build pre-norm encoder layer with GELU activation and 4x MLP expansion
        block = nn.TransformerEncoderLayer(
            dim,
            heads,
            4 * dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Stack depth layers followed by a final LayerNorm wrapper
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
        # Execute encoder stack using key padding mask
        return self.blocks(tokens, src_key_padding_mask=padding)  # (B, N, D)
