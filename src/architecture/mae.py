"""The cross-sensor masked autoencoder, assembled from its parts."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint

from architecture.components.crossencoder import CrossSensorEncoder
from architecture.components.decoder import Decoder
from architecture.components.encoder import Encoder
from architecture.tokens import Tokens

# B = batch, K = patches, S = tokens read together, D = token channels,
# P = patch dimensions.


class CrossSensorMAE(nn.Module):
    """An encoder and a decoder per sensor, one shared encoder aligning them between."""

    def __init__(
        self,
        shapes: dict[str, tuple[int, ...]],
        strides: dict[str, float],
        encoder_dim: int,
        encoder_heads: int,
        encoder_depth: int,
        crossencoder_depth: int,
        decoder_dim: int,
        decoder_heads: int,
        decoder_depth: int,
        attention_m: float,
    ) -> None:
        """Build every part for the instruments the model reads.

        Args:
            shapes: The shape of one patch of each instrument, keyed as ODE names it.
            strides: How far apart two neighbouring patch centres sit, in metres.
            encoder_dim: How wide a token is everywhere but the decoders.
            encoder_heads: How many attention heads every encoder runs.
            encoder_depth: How many blocks each instrument encoder stacks.
            crossencoder_depth: How many blocks the cross-sensor encoder stacks.
            decoder_dim: How wide a token is in the decoders.
            decoder_heads: How many attention heads the decoders run.
            decoder_depth: How many blocks each decoder stacks.
            attention_m: How far apart on the ground two tokens may attend, in metres.
        """
        super().__init__()
        # One encoder per instrument, each from its patches to tokens
        self.encoders = nn.ModuleDict(
            {
                name: Encoder(
                    shape,
                    encoder_dim,
                    encoder_heads,
                    encoder_depth,
                    strides[name],
                    attention_m,
                )
                for name, shape in shapes.items()
            }
        )
        # One stack the instruments read together pass through, on shared weights
        self.crossencoder = CrossSensorEncoder(
            encoder_dim, encoder_heads, crossencoder_depth, attention_m
        )
        # One decoder per instrument, each writing its patches back from the tokens
        self.decoders = nn.ModuleDict(
            {
                name: Decoder(
                    shape,
                    encoder_dim,
                    decoder_dim,
                    decoder_heads,
                    decoder_depth,
                    strides[name],
                    attention_m,
                )
                for name, shape in shapes.items()
            }
        )
        # The token width, which an instrument with no patch is handed zeros at
        self.dim = encoder_dim
        # How far apart on the ground two tokens may attend, in metres
        self.attention_m = attention_m

    def encoded_tokens(
        self, batch: dict[str, Tokens], counted: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """Return each instrument's patches as its own encoder reads them.

        Args:
            batch: Each instrument's patches over the batch.
            counted: Which of its patches the encoders may read. (B, K)

        Returns:
            encoded: Each instrument's tokens, meaningful where counted. (B, K, D)
        """
        encoded = {}
        for name, tokens in batch.items():
            # Guard against empty/absent sensor token inputs
            if tokens.values.shape[1] == 0:
                encoded[name] = tokens.values.new_zeros(
                    tokens.values.shape[0], 0, self.dim
                )  # (B, 0, D)
                continue
            # The instrument's own encoder turns its readable patches into tokens
            encoded[name] = self.encoders[name](
                tokens.values,
                tokens.measured,
                tokens.position,
                counted[name],
            )  # (B, K, D)
        return encoded

    def joint_tokens(
        self,
        encoded: dict[str, Tensor],
        batch: dict[str, Tokens],
        counted: dict[str, Tensor],
        read: list[str],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return some instruments' tokens side by side, attended over together.

        Args:
            encoded: Each instrument's tokens from its own encoder. (B, K, D)
            batch: Each instrument's patches over the batch, which place the tokens.
            counted: Which of its tokens count: visible while training, else present.
            read: Which instruments are read together.

        Returns:
            tokens: The tokens, in the space every instrument shares. (B, S, D)
            position: Where each token's patch sits and reaches, in metres. (B, S, 6)
            counted: Which of them count. (B, S)
        """
        position = torch.cat(
            [batch[name].position for name in read], dim=1
        )  # (B, S, 6)
        readable = torch.cat([counted[name] for name in read], dim=1)  # (B, S)
        tokens = torch.cat([encoded[name] for name in read], dim=1)  # (B, S, D)
        return self.crossencoder(tokens, readable, position), position, readable

    def embed(
        self, batch: dict[str, Tokens], present: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return the tokens standing for each tile, over every instrument it holds.

        Args:
            batch: Each instrument's patches over the batch.
            present: Which slots hold a patch rather than padding. (B, K)

        Returns:
            tokens: Every instrument's tokens side by side, of unit length. (B, S, D)
            position: Where each token's patch sits and reaches, in metres. (B, S, 6)
            present: Which of them hold a patch. (B, S)
        """
        # Every real patch of every instrument is encoded, nothing hidden
        encoded = self.encoded_tokens(batch, present)
        tokens, position, counted = self.joint_tokens(
            encoded, batch, present, list(batch)
        )
        return functional.normalize(tokens, dim=-1), position, counted

    def forward(
        self,
        batch: dict[str, Tokens],
        visible: dict[str, Tensor],
        hidden: dict[str, Tensor],
    ) -> dict[tuple[str, str], Tensor]:
        """Return every instrument's hidden patches, from all and from the others.

        Args:
            batch: Each instrument's patches over the batch.
            visible: Which patches each encoder may read. (B, K)
            hidden: Which patches are predicted, none of them padding. (B, K)

        Returns:
            predictions: Under "umr" each instrument's patches read from every
                instrument, under "cmr" from every other one. (B, K, *P)
        """
        # Only the visible patches are encoded, each instrument on its own
        encoded = self.encoded_tokens(batch, visible)
        # None reads every instrument, a name every instrument but that one
        reads = {None: list(batch)} | {
            name: [one for one in batch if one != name] for name in batch
        }
        joint = {
            left: self.joint_tokens(encoded, batch, visible, read)
            for left, read in reads.items()
        }
        return {
            # Recomputed on the way back, so only one decoder pass is held at once
            (term, name): checkpoint(
                self.decoders[name],
                *joint[left],
                tokens.position,
                hidden[name],
                use_reentrant=False,
            )  # (B, K, *P)
            for name, tokens in batch.items()
            for term, left in (("umr", None), ("cmr", name))
        }
