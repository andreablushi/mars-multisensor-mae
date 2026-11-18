"""Predicting a sensor's hidden patches out of the tokens around them."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from architecture.components.acquisition_conditioning import AcquisitionConditioning
from architecture.components.transformer import Transformer
from architecture.models.context import Context


class Decoder(nn.Module):
    """Let each hidden patch attend over the tokens of its tile, then predict it.

    Attributes:
        shape: The shape of one patch this decoder predicts.
        expand: From the shared width to the decoder width.
        mask: The token standing in for a hidden patch. (D')
        blocks: The transformer the patches read the context through.
        predict: From a token to every sample of its patch, read off the graph.
        acquisition: What tells a hidden patch's query how it was acquired, or None.
        latent: From a token to its patch's target token, I-JEPA's predictor output.
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
        # Predict each patch's target token from its query
        self.latent = nn.Linear(dim, shared)

    def latents_and_patches(
        self,
        context: Tensor,
        context_position: Tensor,
        context_readable: Tensor,
        position: Tensor,
        acquisition: Tensor,
        hidden: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return the hidden patches' target tokens and samples, read from one context.

        Args:
            context: The tokens the prediction reads. (B, S, D)
            context_position: Each one's patch centre, in metres or rows. (B, S, 3)
            context_readable: Which of them count. (B, S)
            position: Each asked patch's centre, in metres or rows. (B, K, 3)
            acquisition: Each asked patch's acquisition angles, in degrees. (B, K, A)
            hidden: Which of them to predict. (B, K)

        Returns:
            latents: One target token per slot, meaningful where hidden. (B, K, D)
            patches: One patch per slot, written without a gradient to the tokens.
                (B, K, *P)
        """
        # One mask token per slot as a query
        queries = self.mask.expand(*hidden.shape, -1)  # (B, K, D')
        if self.acquisition is not None:
            queries = self.acquisition(queries, acquisition)
        keys = torch.cat([self.expand(context), queries], dim=1)  # (B, S + K, D')
        placed = torch.cat([context_position, position], dim=1)  # (B, S + K, 3)
        # The readable context and the hidden patches are read, nothing else
        readable = torch.cat([context_readable, hidden], dim=1)  # (B, S + K)
        # The context is only keys, only the patch slots ask as queries and are updated
        keys = self.blocks(keys, readable.unsqueeze(1), placed, context.shape[1])
        queries = keys[:, context.shape[1] :]  # (B, K, D')
        # Write every sample of each patch slot from its query, off the graph
        written = self.predict(queries.detach())  # (B, K, prod P)
        return self.latent(queries), written.unflatten(-1, self.shape)

    def forward(
        self,
        own: Context,
        others: Context,
        position: Tensor,
        acquisition: Tensor,
        hidden: Tensor,
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """Return the instrument's hidden patches, from its own tokens and the others'.

        Args:
            own: The instrument's own tokens.
            others: Every other instrument's tokens.
            position: Each asked patch's centre, in metres or rows. (B, K, 3)
            acquisition: Each asked patch's acquisition angles, in degrees. (B, K, A)
            hidden: Which of them to predict. (B, K)

        Returns:
            umr: The target tokens and patches read from the instrument's own tokens.
                (B, K, D), (B, K, *P)
            cmr: The same read from every other instrument's. (B, K, D), (B, K, *P)
        """
        # Each pass is recomputed on the way back, so only one is held at once
        umr = checkpoint(
            self.latents_and_patches,
            *own,
            position,
            acquisition,
            hidden,
            use_reentrant=False,
        )
        cmr = checkpoint(
            self.latents_and_patches,
            *others,
            position,
            acquisition,
            hidden,
            use_reentrant=False,
        )
        return umr, cmr
