"""Writing a sensor's hidden patches back out of the tokens around them."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from architecture.components.span_encoding import SpanEncoding
from architecture.components.transformer import Transformer

# B = batch, S = context tokens, K = target patches, D = token channels,
# P = patch dimensions.


class Decoder(nn.Module):
    """Let each hidden patch attend over the tokens of its tile, then write it.

    Attributes:
        shape: The shape of one patch this decoder predicts.
        expand: From the shared width to the decoder width.
        mask: The token standing in for a hidden patch. (D')
        span: The encoding of how far each patch reaches, at the patch spacing.
        blocks: The transformer the patches read the context through.
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
        # Bring tokens from the shared width to the decoder width
        self.expand = nn.Linear(shared, dim)
        # One learned token standing in for every hidden patch
        self.mask = nn.Parameter(torch.zeros(dim))  # (D')
        # Start the mask token small, as transformer embeddings are
        nn.init.normal_(self.mask, std=0.02)
        # Encode how far every token's patch reaches, at the patch spacing
        self.span = SpanEncoding(dim, stride)
        # The blocks the patches read the context through
        self.blocks = Transformer(dim, heads, depth)
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
            context: The tokens the prediction reads, of any sensor. (B, S, D)
            context_position: Each one's centre and span, in metres or rows. (B, S, 6)
            context_visible: Which of them hold anything. (B, S)
            position: Each asked patch's centre and span, in metres or rows. (B, K, 6)
            hidden: Which of them to predict. (B, K)

        Returns:
            prediction: One patch per slot, meaningful where hidden. (B, K, *P)
        """
        # The context at the decoder width, then one mask token per slot as a query
        keys = torch.cat(
            [
                self.expand(context),
                self.mask.expand(*position.shape[:2], -1),
            ],
            dim=1,
        )  # (B, S + K, D')
        placed = torch.cat([context_position, position], dim=1)  # (B, S + K, 6)
        # The visible context and the hidden patches are read, nothing else
        readable = torch.cat([context_visible, hidden], dim=1)  # (B, S + K)
        # The context is only keys, only the patch slots ask as queries and are updated
        keys = self.blocks(keys + self.span(placed), readable, placed, context.shape[1])
        queries = keys[:, context.shape[1] :]  # (B, K, D')
        # Write every sample of each patch slot from its query
        written = self.predict(queries)  # (B, K, prod P)
        return written.unflatten(-1, self.shape)  # (B, K, *P)
