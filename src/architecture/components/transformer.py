"""Attending over a set of tokens, each over the ones that carry something."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional

from architecture.components.rotary_encoding import RotaryEncoding

# B = batch, N = tokens, D = token channels.


class Block(nn.Module):
    """One pre-norm block: masked self-attention, then a feed-forward.

    Attributes:
        heads: How many attention heads the block runs.
        attend_norm: What the tokens are normalised by before they attend.
        qkv: From a token to its query, key and value.
        rotate: The rotary encoding queries and keys are turned by.
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
        self.rotate = RotaryEncoding(dim // heads)
        self.out = nn.Linear(dim, dim)
        self.feed_norm = nn.LayerNorm(dim)
        self.feed = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(
        self, keys: Tensor, mask: Tensor, position: Tensor, fixed: int
    ) -> Tensor:
        """Return the keys after one block, the queries among them updated.

        Args:
            keys: Every token, each read as a key and a value, those past the fixed
                also asking as queries. (B, N, D)
            mask: Which key each query may attend to. (B, 1, N - fixed, N)
            position: Each token's patch centre and span, in metres or rows. (B, N, 6)
            fixed: How many leading tokens are keys alone, never queries.

        Returns:
            keys: The keys, the queries among them updated. (B, N, D)
        """
        # Queries, keys and values split into heads
        query, key, value = (
            self.qkv(self.attend_norm(keys))
            .unflatten(-1, (3, self.heads, -1))
            .permute(2, 0, 3, 1, 4)
        )  # (B, H, N, D / H) each
        # Queries and keys turned by where they sit, so a score reads their offset
        answered = functional.scaled_dot_product_attention(
            self.rotate(query[..., fixed:, :], position[:, fixed:]),
            self.rotate(key, position),
            value,
            attn_mask=mask,
        )  # (B, H, N - fixed, D / H)
        queries = keys[:, fixed:] + self.out(answered.transpose(1, 2).flatten(2))
        queries = queries + self.feed(self.feed_norm(queries))  # (B, N - fixed, D)
        return torch.cat([keys[:, :fixed], queries], dim=1)  # (B, N, D)


class Transformer(nn.Module):
    """Pre-norm blocks of attention stacked, normalised once more at the end.

    Attributes:
        blocks: The blocks.
        norm: The last normalisation.
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

    def forward(
        self, keys: Tensor, readable: Tensor, position: Tensor, fixed: int = 0
    ) -> Tensor:
        """Return the keys after the queries among them attend over the readable ones.

        Args:
            keys: Every token, each read as a key and a value, those past the fixed
                also asking as queries. (B, N, D)
            readable: Which keys carry something. (B, N)
            position: Each token's patch centre and span, in metres or rows. (B, N, 6)
            fixed: How many leading tokens are keys alone, never queries.

        Returns:
            keys: The keys, the queries among them attended. (B, N, D)
        """
        # A token always reads itself, so a row with no other key stays finite
        itself = torch.eye(keys.shape[1], dtype=torch.bool, device=keys.device)[
            fixed:
        ]  # (N', N)
        mask = (readable.unsqueeze(1) | itself).unsqueeze(1)  # (B, 1, N', N)
        for block in self.blocks:
            keys = block(keys, mask, position, fixed)
        return self.norm(keys)  # (B, N, D)
