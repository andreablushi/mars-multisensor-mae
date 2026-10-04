"""Attending over a set of tokens, each over the ones near it that carry something."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional

# B = batch, N = tokens, D = token channels.


class Block(nn.Module):
    """One pre-norm block: masked self-attention, then a feed-forward.

    Attributes:
        heads: How many attention heads the block runs.
        attend_norm: What the tokens are normalised by before they attend.
        qkv: From a token to its query, key and value.
        out: From the attended heads back to the token width.
        feed_norm: What the tokens are normalised by before the feed-forward.
        feed: The feed-forward, four times as wide inside.
    """

    def __init__(self, dim: int, heads: int) -> None:
        """Build one block for one token width.

        Args:
            dim: The token width.
            heads: How many attention heads the block runs.
        """
        super().__init__()
        self.heads = heads
        self.attend_norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.feed_norm = nn.LayerNorm(dim)
        self.feed = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
        """Return the tokens after one block.

        Args:
            tokens: The tokens. (B, N, D)
            mask: Which key each token may attend to. (B, 1, N, N)

        Returns:
            tokens: The updated tokens. (B, N, D)
        """
        # Queries, keys and values split into heads
        query, key, value = (
            self.qkv(self.attend_norm(tokens))
            .unflatten(-1, (3, self.heads, -1))
            .permute(2, 0, 3, 1, 4)
        )  # (B, H, N, D / H) each
        attended = functional.scaled_dot_product_attention(
            query, key, value, attn_mask=mask
        )  # (B, H, N, D / H)
        tokens = tokens + self.out(attended.transpose(1, 2).flatten(2))  # (B, N, D)
        return tokens + self.feed(self.feed_norm(tokens))  # (B, N, D)


class Transformer(nn.Module):
    """Pre-norm blocks of local attention stacked, normalised once more at the end.

    Attributes:
        radius: How far apart on the ground two tokens may attend, in metres.
        blocks: The blocks.
        norm: The last normalisation.
    """

    def __init__(self, dim: int, heads: int, depth: int, radius: float) -> None:
        """Build the stack for one token width.

        Args:
            dim: The token width.
            heads: How many attention heads each block runs.
            depth: How many blocks are stacked.
            radius: How far apart on the ground two tokens may attend, in metres.
        """
        super().__init__()
        self.radius = radius
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: Tensor, attended: Tensor, position: Tensor) -> Tensor:
        """Return the tokens after attending over the near ones that carry something.

        Args:
            tokens: The tokens to attend over. (B, N, D)
            attended: Which of them carry something. (B, N)
            position: Where each token's patch sits and reaches, in metres. (B, N, 6)

        Returns:
            tokens: The attended tokens. (B, N, D)
        """
        ground = position[..., :2]  # (B, N, 2)
        near = torch.cdist(ground, ground) <= self.radius  # (B, N, N)
        # A token always reads itself, so a row with no other key stays finite
        itself = torch.eye(near.shape[-1], dtype=torch.bool, device=near.device)
        mask = ((near & attended.unsqueeze(1)) | itself).unsqueeze(1)  # (B, 1, N, N)
        for block in self.blocks:
            tokens = block(tokens, mask)
        return self.norm(tokens)  # (B, N, D)
