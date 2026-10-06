"""The cross-sensor masked autoencoder, assembled from its parts."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional

from architecture.components.crossencoder import CrossSensorEncoder
from architecture.components.decoder import Decoder
from architecture.components.encoder import Encoder
from architecture.tokens import Context, Tokens

# B = batch, K = patches, S = tokens read together, D = token channels,
# P = patch dimensions.


class CrossSensorMAE(nn.Module):
    """An encoder and a decoder per sensor, one shared encoder aligning them between."""

    def __init__(
        self,
        shapes: dict[str, tuple[int, ...]],
        encoder_dim: int,
        encoder_heads: int,
        encoder_depth: int,
        crossencoder_depth: int,
        decoder_dim: int,
        decoder_heads: int,
        decoder_depth: int,
    ) -> None:
        """Build every part for the instruments the model reads.

        Args:
            shapes: The shape of one patch of each instrument, keyed as ODE names it.
            encoder_dim: How wide a token is everywhere but the decoders.
            encoder_heads: How many attention heads every encoder runs.
            encoder_depth: How many blocks each instrument encoder stacks.
            crossencoder_depth: How many blocks the cross-sensor encoder stacks.
            decoder_dim: How wide a token is in the decoders.
            decoder_heads: How many attention heads the decoders run.
            decoder_depth: How many blocks each decoder stacks.
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
                )
                for name, shape in shapes.items()
            }
        )
        # One stack the instruments read together pass through, on shared weights
        self.crossencoder = CrossSensorEncoder(
            encoder_dim, encoder_heads, crossencoder_depth
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
                )
                for name, shape in shapes.items()
            }
        )

    def shared_tokens(
        self, batch: dict[str, Tokens], readable: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """Return each instrument's tokens in the shared space, each read on its own.

        Args:
            batch: Each instrument's patches over the batch.
            readable: Which patches each encoder may read. (B, K)

        Returns:
            tokens: Each instrument's tokens, meaningful where readable. (B, K, D)
        """
        # The shared weights see one instrument at a time, so none reads another
        return {
            name: self.crossencoder(
                self.encoders[name](one, readable[name]), readable[name], one.position
            )
            for name, one in batch.items()
        }

    @staticmethod
    def context(
        tokens: dict[str, Tensor],
        batch: dict[str, Tokens],
        readable: dict[str, Tensor],
        read: list[str],
    ) -> Context:
        """Return some instruments' tokens side by side, as a decoder reads them.

        Args:
            tokens: Each instrument's tokens in the shared space. (B, K, D)
            batch: Each instrument's patches over the batch, which place the tokens.
            readable: Which of each instrument's tokens count. (B, K)
            read: Which instruments are put side by side.

        Returns:
            context: Their tokens, centres and which of them count.
        """
        return Context(
            torch.cat([tokens[name] for name in read], 1),  # (B, S, D)
            torch.cat([batch[name].position for name in read], 1),  # (B, S, 3)
            torch.cat([readable[name] for name in read], 1),  # (B, S)
        )

    def embed(self, batch: dict[str, Tokens], present: dict[str, Tensor]) -> Context:
        """Return the tokens standing for each tile, over every instrument it holds.

        Args:
            batch: Each instrument's patches over the batch.
            present: Which slots hold a patch rather than padding. (B, K)

        Returns:
            context: Every instrument's tokens side by side, of unit length, their
                centres and which of them hold a patch.
        """
        tokens, position, present = self.context(
            self.shared_tokens(batch, present), batch, present, list(batch)
        )
        return Context(functional.normalize(tokens, dim=-1), position, present)

    def forward(
        self,
        batch: dict[str, Tokens],
        visible: dict[str, Tensor],
        hidden: dict[str, Tensor],
    ) -> dict[str, dict[str, Tensor]]:
        """Return every instrument's hidden patches, from its own tokens and others'.

        Args:
            batch: Each instrument's patches over the batch.
            visible: Which patches each encoder may read. (B, K)
            hidden: Which patches are predicted, none of them padding. (B, K)

        Returns:
            predictions: Under "umr" each instrument's patches read from its own
                tokens, under "cmr" from every other instrument's. (B, K, *P)
        """
        tokens = self.shared_tokens(batch, visible)
        predictions = {"umr": {}, "cmr": {}}
        for name, one in batch.items():
            others = [other for other in batch if other != name]
            predictions["umr"][name], predictions["cmr"][name] = self.decoders[name](
                self.context(tokens, batch, visible, [name]),
                self.context(tokens, batch, visible, others),
                one.position,
                hidden[name],
            )  # (B, K, *P) each
        return predictions
