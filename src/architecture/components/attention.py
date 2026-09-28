"""Attending from one set of tokens to another, by what they hold and where they sit."""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional

from architecture.components.rotary_encoding import RotaryEncoding

# B = batch, H = heads, Q = asking tokens, S = answering tokens, D = token channels,
# E = head channels.


class Attention(nn.Module):
    """Multi-head attention whose queries and keys are rotated by where tokens sit.

    Attributes:
        heads: How many attention heads it runs.
        query: From an asking token to its queries.
        key_value: From an answering token to its keys and values.
        out: From the heads back to one token.
        rotary: The rotary encoding.
    """

    def __init__(self, dim: int, heads: int) -> None:
        """Build the attention for one token width.

        Args:
            dim: The token width.
            heads: How many attention heads it runs.
        """
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(dim, dim)
        self.key_value = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, dim)
        self.rotary = RotaryEncoding(dim // heads)

    def forward(
        self,
        asking: Tensor,
        asking_position: Tensor,
        answering: Tensor,
        answering_position: Tensor,
        allowed: Tensor,
    ) -> Tensor:
        """Return what each asking token reads from the answering ones it may read.

        Args:
            asking: The tokens asking. (B, Q, D)
            asking_position: Where each sits and reaches. (B, Q, 6)
            answering: The tokens answering. (B, S, D)
            answering_position: Where each sits and reaches. (B, S, 6)
            allowed: Whether each asking token may read each answering one. (B, Q|1, S)

        Returns:
            read: One token per asking token. (B, Q, D)
        """
        query = self.query(asking).unflatten(-1, (self.heads, -1)).transpose(1, 2)
        key, value = (
            self.key_value(answering)
            .unflatten(-1, (2, self.heads, -1))
            .permute(2, 0, 3, 1, 4)
        )  # (B, H, S, E) each
        read = functional.scaled_dot_product_attention(
            self.rotary(query, asking_position),
            self.rotary(key, answering_position),
            value,
            attn_mask=allowed.unsqueeze(1),
        )  # (B, H, Q, E)
        return self.out(read.transpose(1, 2).flatten(2))  # (B, Q, D)
