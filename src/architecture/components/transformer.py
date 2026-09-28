"""Attending over a set of tokens, skipping the ones that carry nothing."""

from __future__ import annotations

from torch import Tensor, nn

from architecture.components.attention import Attention

# B = batch, N = tokens, D = token channels.


class Block(nn.Module):
    """One pre-norm transformer block, its attention placed by the rotary encoding.

    Attributes:
        attention_norm: What a token is normalised by before it attends.
        attend: The attention.
        mlp_norm: What a token is normalised by before the MLP.
        mlp: The MLP.
    """

    def __init__(self, dim: int, heads: int) -> None:
        """Build the block for one token width.

        Args:
            dim: The token width.
            heads: How many attention heads it runs.
        """
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attend = Attention(dim, heads)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, tokens: Tensor, position: Tensor, allowed: Tensor) -> Tensor:
        """Return the tokens after one round of attention and MLP.

        Args:
            tokens: The tokens. (B, N, D)
            position: Where each sits and reaches. (B, N, 6)
            allowed: Which tokens may be read. (B, 1, N)

        Returns:
            tokens: The updated tokens. (B, N, D)
        """
        normed = self.attention_norm(tokens)
        tokens = tokens + self.attend(normed, position, normed, position, allowed)
        return tokens + self.mlp(self.mlp_norm(tokens))  # (B, N, D)


class Transformer(nn.Module):
    """Pre-norm transformer blocks stacked, normalised once more at the end.

    Attributes:
        blocks: The blocks.
        norm: What the tokens are normalised by at the end.
    """

    def __init__(self, dim: int, heads: int, depth: int) -> None:
        """Build the stack for one token width.

        Args:
            dim: The token width.
            heads: How many attention heads each block runs.
            depth: How many blocks are stacked.
        """
        super().__init__()
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: Tensor, position: Tensor, attended: Tensor) -> Tensor:
        """Return the tokens after attending over the ones that carry something.

        Args:
            tokens: The tokens to attend over. (B, N, D)
            position: Where each sits and reaches. (B, N, 6)
            attended: Which of them carry something. (B, N)

        Returns:
            tokens: The attended tokens. (B, N, D)
        """
        allowed = attended.clone()  # (B, N)
        # Unmask wholly empty sequences to keep the soft-max from going nan
        allowed[~allowed.any(dim=1)] = True
        for block in self.blocks:
            tokens = block(tokens, position, allowed.unsqueeze(1))  # (B, N, D)
        return self.norm(tokens)  # (B, N, D)
