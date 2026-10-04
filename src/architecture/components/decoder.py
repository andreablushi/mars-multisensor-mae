"""Writing a sensor's hidden patches back out of the cells around them."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from architecture.components.positional_encoding import PositionalEncoding
from architecture.grid import neighbourhoods, relative_boxes

# B = batch, C = cells, K = target patches, N = hidden patches, M = window cells,
# D = token channels, P = patch dimensions.

PATCHES = 1024

REACH = 3.0


class CrossAttentionBlock(nn.Module):
    """One pre-norm block: the patch attends over its cells, then a feed-forward.

    Attributes:
        attend_norm: What the patch is normalised by before it attends.
        attend: The attention from the patch to its cells.
        feed_norm: What the patch is normalised by before the feed-forward.
        feed: The feed-forward, four times as wide inside.
    """

    def __init__(self, dim: int, heads: int) -> None:
        """Build one block for one token width.

        Args:
            dim: The token width.
            heads: How many attention heads the block runs.
        """
        super().__init__()
        # Normalise the patch before it attends
        self.attend_norm = nn.LayerNorm(dim)
        # The patch reads its cells
        self.attend = nn.MultiheadAttention(dim, heads, batch_first=True)
        # Normalise the patch before the feed-forward
        self.feed_norm = nn.LayerNorm(dim)
        # The feed-forward, four times as wide inside
        self.feed = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, query: Tensor, cells: Tensor, ignored: Tensor) -> Tensor:
        """Return the patch token after it reads its cells.

        Args:
            query: The patch token. (N, 1, D)
            cells: The cells it reads. (N, M, D)
            ignored: The padding attention skips. (N, M)

        Returns:
            query: The updated patch token. (N, 1, D)
        """
        # The patch attends over its cells, padding skipped, and adds what it read
        read, _ = self.attend(
            self.attend_norm(query),
            cells,
            cells,
            key_padding_mask=ignored,
            need_weights=False,
        )
        query = query + read
        # The feed-forward, added back
        return query + self.feed(self.feed_norm(query))  # (N, 1, D)


class Decoder(nn.Module):
    """Let each hidden patch attend over the cells around it, then write it.

    A patch reads the occupied cells within one patch of its own size on every
    side, each placed by where it sits against the patch.

    Attributes:
        shape: The shape of one patch this decoder predicts.
        expand: From the shared width up to the decoder width, at unit scale.
        mask: The token standing in for a hidden patch. (D')
        place: The positional encoding of a cell's offset from the patch.
        blocks: The cross-attention blocks, from the patch to its cells.
        norm: What the patch is normalised by after the blocks.
        predict: From a token to every sample of its patch.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        shared: int,
        dim: int,
        heads: int,
        depth: int,
        stride: float,
    ) -> None:
        """Build the decoder for one instrument.

        Args:
            shape: The shape of one patch of the instrument.
            shared: The width the cross-sensor encoder hands tokens at.
            dim: The decoder's token width.
            heads: How many attention heads each block runs.
            depth: How many blocks are stacked.
            stride: How far apart two neighbouring patch centres sit, in metres.
        """
        super().__init__()
        # The shape of one patch, which the output is unflattened to
        self.shape = shape
        # Bring cells from the shared width to the decoder width, normalised
        self.expand = nn.Sequential(nn.Linear(shared, dim), nn.LayerNorm(dim))
        # One learned token standing in for every hidden patch
        self.mask = nn.Parameter(torch.zeros(dim))  # (D')
        # Start the mask token small, as transformer embeddings are
        nn.init.normal_(self.mask, std=0.02)
        # Encode a cell's offset from the patch, at the instrument's patch spacing
        self.place = PositionalEncoding(dim, stride)
        # The blocks the patch reads its cells through
        self.blocks = nn.ModuleList(
            CrossAttentionBlock(dim, heads) for _ in range(depth)
        )
        # Normalise once more at the end
        self.norm = nn.LayerNorm(dim)
        # Write every sample of the patch from its token
        self.predict = nn.Linear(dim, math.prod(shape))
        # Start by predicting zero, the dataset's mean after standardisation
        nn.init.zeros_(self.predict.weight)
        nn.init.zeros_(self.predict.bias)

    def forward(
        self,
        context: Tensor,
        context_position: Tensor,
        context_visible: Tensor,
        position: Tensor,
        hidden: Tensor,
    ) -> Tensor:
        """Return the predicted values of the hidden patches.

        Args:
            context: The cells the prediction reads, of any sensor. (B, C, D)
            context_position: Where each sits and reaches, in metres. (B, C, 6)
            context_visible: Which of them hold anything. (B, C)
            position: Where each patch asked for sits and reaches, in metres. (B, K, 6)
            hidden: Which of them to predict. (B, K)

        Returns:
            prediction: One patch per slot, meaningful where hidden. (B, K, *P)
        """
        # Every slot starts at zero, and only hidden ones are written
        prediction = position.new_zeros(*position.shape[:2], *self.shape)
        # The tile and slot of every hidden patch
        tiles, slot = hidden.nonzero(as_tuple=True)  # (N,), (N,)
        # With nothing hidden there is nothing to write
        if not len(tiles):
            return prediction  # (B, K, *P)
        # The cells at the decoder width
        cells = self.expand(context)  # (B, C, D')
        # Where each hidden patch sits and reaches
        asked = position[tiles, slot]  # (N, 6)
        # A patch's window is its own box, every span tripled
        window = torch.cat([asked[:, :3], asked[:, 3:] * REACH], dim=-1)  # (N, 6)
        written = []
        # Read the hidden patches a chunk at a time, so their windows fit in memory
        for start in range(0, len(tiles), PATCHES):
            # The hidden patches of this chunk
            rows = slice(start, start + PATCHES)
            # The tile each patch of the chunk belongs to
            tile = tiles[rows]  # (n,)
            # The cells in each patch's window, padded to one count
            at, chosen, ignored = neighbourhoods(
                window[rows], context_position[tile], context_visible[tile]
            )  # (n, M)
            # Each cell's centre relative to the patch's, beside its own spans
            offset = relative_boxes(context_position[tile[:, None], at], asked[rows])
            # Each cell's vector, placed by its offset from the patch
            read = cells[tile[:, None], at] + self.place(offset)  # (n, M, D')
            # A window holding no cell reads zeros, so it writes a learned constant
            read = read * chosen.unsqueeze(-1)  # (n, M, D')
            # Every hidden patch asks with the same mask token
            decoded = self.mask.expand(len(tile), 1, -1)  # (n, 1, D')
            # The patch reads the cells of its window, block after block
            for block in self.blocks:
                decoded = block(decoded, read, ignored)
            # Write every sample of each patch from its decoded token
            written.append(self.predict(self.norm(decoded[:, 0])))  # (n, prod P)
        # Put the written patches back into their slots, in the patch's own shape
        prediction[tiles, slot] = (
            torch.cat(written).unflatten(-1, self.shape).to(prediction.dtype)
        )
        return prediction  # (B, K, *P)
