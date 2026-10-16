"""The cross-sensor masked autoencoder, assembled from its parts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from architecture.components.crossencoder import CrossSensorEncoder
from architecture.components.decoder import Decoder
from architecture.components.encoder import Encoder
from architecture.tokens import Context, Tokens

# B = batch, K = patches, S = tokens read together, D = token channels,
# P = patch dimensions.


class CrossSensorMAE(nn.Module):
    """An encoder and a decoder per sensor, one shared encoder aligning them between.

    The shared encoder also hands back one CLS of every sensor, which every decoder
    reads, so the whole loss trains it.
    """

    def __init__(
        self,
        shapes: dict[str, tuple[int, ...]],
        conditions: dict[str, int],
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
            conditions: How many values each instrument's patches are conditioned on.
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
                    shape, conditions[name], encoder_dim, encoder_heads, encoder_depth
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
                    conditions[name],
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
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return the CLS of every instrument, and each one's tokens in shared space.

        Args:
            batch: Each instrument's patches over the batch.
            readable: Which patches each encoder may read. (B, K)

        Returns:
            cls: The token read over every instrument together. (B, D)
            tokens: Each instrument's CLS token, then its tokens, meaningful where
                readable. (B, 1 + K, D)
        """
        led = [one.led_by_cls(readable[name]) for name, one in batch.items()]
        cls, tokens = self.crossencoder(
            [self.encoders[name](one, readable[name]) for name, one in batch.items()],
            [counted for counted, _ in led],
            [position for _, position in led],
        )
        return cls, dict(zip(batch, tokens, strict=True))

    @staticmethod
    def context(
        cls: Tensor,
        tokens: dict[str, Tensor],
        batch: dict[str, Tokens],
        readable: dict[str, Tensor],
        read: list[str],
    ) -> Context:
        """Return some instruments' tokens side by side, as a decoder reads them.

        Args:
            cls: The token read over every instrument together. (B, D)
            tokens: Each instrument's CLS token and tokens in the shared space.
                (B, 1 + K, D)
            batch: Each instrument's patches over the batch, which place the tokens.
            readable: Which of each instrument's patch tokens count. (B, K)
            read: Which instruments are put side by side.

        Returns:
            context: Their tokens, centres and which of them count, led by the CLS
                of every instrument at the tile centre, their own CLS tokens too.
        """
        led = [batch[name].led_by_cls(readable[name]) for name in read]
        return Context(
            torch.cat([cls.unsqueeze(1), *(tokens[name] for name in read)], 1),
            torch.cat([cls.new_zeros(len(cls), 1, 3), *(at for _, at in led)], 1),
            torch.cat(
                [cls.new_ones(len(cls), 1, dtype=torch.bool), *(on for on, _ in led)], 1
            ),  # (B, S)
        )

    def forward(
        self,
        batch: dict[str, Tokens],
        visible: dict[str, Tensor],
        hidden: dict[str, Tensor],
        sources: Mapping[str, Sequence[str]],
    ) -> dict[str, dict[str, Tensor]]:
        """Return every instrument's hidden patches, from its own tokens and others'.

        Args:
            batch: Each instrument's patches over the batch.
            visible: Which patches each encoder may read. (B, K)
            hidden: Which patches are predicted, none of them padding. (B, K)
            sources: The instruments each instrument is rebuilt from cross-modally.

        Returns:
            predictions: Under "umr" each instrument's patches read from its own
                tokens, under "cmr" from its sources' tokens, both with the CLS of
                every instrument. (B, K, *P)
        """
        read = {name: one.kept(visible[name]) for name, one in batch.items()}
        seen = {name: kept for name, (kept, _) in read.items()}
        readable = {name: counted for name, (_, counted) in read.items()}
        cls, tokens = self.shared_tokens(seen, readable)
        predictions = {"umr": {}, "cmr": {}}
        for name, one in batch.items():
            predictions["umr"][name], predictions["cmr"][name] = self.decoders[name](
                self.context(cls, tokens, seen, readable, [name]),
                self.context(cls, tokens, seen, readable, sources[name]),
                one.position,
                hidden[name],
                one.condition,
            )  # (B, K, *P) each
        return predictions
