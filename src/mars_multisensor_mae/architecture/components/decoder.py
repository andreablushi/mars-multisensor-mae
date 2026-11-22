"""Predicting a sensor's hidden patches out of the tokens around them."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from mars_multisensor_mae.architecture.components.acquisition_conditioning import (
    AcquisitionConditioning,
)
from mars_multisensor_mae.architecture.components.transformer import Transformer
from mars_multisensor_mae.architecture.models.context import Context


class Decoder(nn.Module):
    """Let each hidden patch attend over the tokens of its tile, then predict it.

    Attributes:
        shape: The shape of one patch this decoder predicts.
        expand: From the shared width to the decoder width.
        mask: The token standing in for a hidden patch. (D')
        blocks: The transformer the patches read the context through.
        predict: From a token to every sample of its patch, read off the graph.
        acquisition: What tells a hidden patch's query how it was acquired, or None.
        latent: From a token to its patch's target latent.
        target: The frozen random projection from a patch to its target latent.
            (prod P, D)
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        acquired: int,
        shared: int,
        dim: int,
        heads: int,
        depth: int,
    ) -> None:
        """Build the decoder for one instrument.

        Args:
            shape: The shape of one patch of the instrument.
            acquired: How many acquisition angles each patch carries.
            shared: The width the cross-sensor encoder hands tokens at.
            dim: The decoder's token width.
            heads: How many attention heads each block runs.
            depth: How many blocks are stacked.
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
        # The blocks the patches read the context through
        self.blocks = Transformer(dim, heads, depth)
        # Write every sample of the patch from its token
        self.predict = nn.Linear(dim, math.prod(shape))
        # Start by predicting zero, the dataset's mean after standardisation
        nn.init.zeros_(self.predict.weight)
        nn.init.zeros_(self.predict.bias)
        self.acquisition = AcquisitionConditioning(acquired, dim) if acquired else None
        # Predict each patch's target latent from its token
        self.latent = nn.Linear(dim, shared)
        # OlmoEarth's Latent MIM Lite target, drawn once and never trained
        self.register_buffer("target", torch.randn(math.prod(shape), shared))

    def forward(
        self, context: Context, position: Tensor, acquisition: Tensor, hidden: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return the hidden patches' latents and samples as read from one context.

        Args:
            context: The tokens the prediction reads. (B, S, D)
            position: Each asked patch's centre, in metres or rows. (B, K, 3)
            acquisition: Each asked patch's acquisition angles, in degrees. (B, K, A)
            hidden: Which of them to predict. (B, K)

        Returns:
            latents: One target latent per slot, meaningful where hidden. (B, K, D)
            patches: One patch per slot, written without a gradient to the tokens.
                (B, K, *P)
        """
        # One mask token per slot as a query
        queries = self.mask.expand(*hidden.shape, -1)  # (B, K, D')
        if self.acquisition is not None:
            queries = self.acquisition(queries, acquisition)
        keys = torch.cat(
            [self.expand(context.tokens), queries], dim=1
        )  # (B, S + K, D')
        placed = torch.cat([context.position, position], dim=1)  # (B, S + K, 3)
        # The readable context and the hidden patches are read, nothing else
        readable = torch.cat([context.readable, hidden], dim=1)  # (B, S + K)
        # The context is only keys, only the patch slots ask as queries and are updated
        # The blocks are recomputed on the way back, so only one pass is held at once
        count = context.tokens.shape[1]
        keys = checkpoint(
            self.blocks,
            keys,
            readable.unsqueeze(1),
            placed,
            count,
            use_reentrant=False,
        )  # (B, S + K, D')
        queries = keys[:, count:]  # (B, K, D')
        # Write every sample of each patch slot from its query, off the graph
        written = self.predict(queries.detach())  # (B, K, prod P)
        return self.latent(queries), written.unflatten(-1, self.shape)
