"""One instrument's encoder, a transformer over the patches it is shown."""

from __future__ import annotations

from torch import Tensor, nn

from architecture.components.embed import PatchEmbedding
from architecture.components.fourier import PositionEncoding


def transformer(dim: int, heads: int, depth: int) -> nn.TransformerEncoder:
    """Return a stack of pre-norm transformer blocks, normalised once more at the end.

    Args:
        dim: The token width.
        heads: How many attention heads each block runs.
        depth: How many blocks are stacked.

    Returns:
        blocks: The stack, read with a key padding mask over the tokens to skip.
    """
    block = nn.TransformerEncoderLayer(
        dim,
        heads,
        4 * dim,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(
        block, depth, norm=nn.LayerNorm(dim), enable_nested_tensor=False
    )


def key_padding(attended: Tensor) -> Tensor:
    """Return which keys every query skips, never every key of one row.

    Args:
        attended: Which tokens carry something to attend to. (B, N)

    Returns:
        padding: True where a key is skipped. A row with nothing to attend to
            skips none, so it comes out finite and is dropped downstream. (B, N)
    """
    padding = ~attended  # (B, N)
    padding[padding.all(dim=1)] = False
    return padding


class Encoder(nn.Module):
    """One instrument's encoder: embed each patch, place it, attend over the visible.

    Attributes:
        embed: The patch embedding.
        place: The position encoding.
        blocks: The transformer.
    """

    def __init__(
        self, shape: tuple[int, ...], dim: int, heads: int, depth: int
    ) -> None:
        """Build the encoder for one instrument.

        Args:
            shape: The shape of one patch of the instrument.
            dim: The token width.
            heads: How many attention heads each block runs.
            depth: How many blocks are stacked.
        """
        super().__init__()
        self.embed = PatchEmbedding(shape, dim)
        self.place = PositionEncoding(dim)
        self.blocks = transformer(dim, heads, depth)

    def forward(self, values: Tensor, position: Tensor, visible: Tensor) -> Tensor:
        """Return the encoded tokens.

        Args:
            values: The normalised patches. (B, K, *P)
            position: Where each patch centre sits, in metres. (B, K, 3)
            visible: Which patches the encoder may read. (B, K)

        Returns:
            tokens: One per slot, meaningful where visible. (B, K, D)
        """
        tokens = self.embed(values) + self.place(position)  # (B, K, D)
        return self.blocks(
            tokens, src_key_padding_mask=key_padding(visible)
        )  # (B, K, D)
