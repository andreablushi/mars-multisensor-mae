"""The cross-sensor masked autoencoder, assembled from its parts."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional

from architecture.components.crossencoder import CrossSensorEncoder
from architecture.components.decoder import Decoder
from architecture.components.encoder import Encoder
from architecture.tokens import Context, Tokens
from configs.load import named_seed
from evaluation.similarity import CLS, global_average_pooling

# B = batch, K = patches, S = tokens read together, D = token channels,
# P = patch dimensions.


class CrossSensorMAE(nn.Module):
    """An encoder and a decoder per sensor, one shared encoder aligning them between.

    The shared encoder also hands back one CLS of every sensor, which every decoder
    reads beside its own tokens, so the UMR loss trains it.
    """

    def __init__(
        self,
        shapes: dict[str, tuple[int, ...]],
        encoder_conditions: dict[str, int],
        decoder_conditions: dict[str, int],
        encoder_dim: int,
        encoder_heads: int,
        encoder_depth: int,
        crossencoder_depth: int,
        decoder_dim: int,
        decoder_heads: int,
        decoder_depth: int,
        seed: int,
    ) -> None:
        """Build every part for the instruments the model reads, each from its own seed.

        Args:
            shapes: The shape of one patch of each instrument, keyed as ODE names it.
            encoder_conditions: How many values each instrument's encoder reads its
                patches conditioned on.
            decoder_conditions: How many values each instrument's decoder reads its
                patches conditioned on.
            encoder_dim: How wide a token is everywhere but the decoders.
            encoder_heads: How many attention heads every encoder runs.
            encoder_depth: How many blocks each instrument encoder stacks.
            crossencoder_depth: How many blocks the cross-sensor encoder stacks.
            decoder_dim: How wide a token is in the decoders.
            decoder_heads: How many attention heads the decoders run.
            decoder_depth: How many blocks each decoder stacks.
            seed: What every part's starting weights are drawn from, so a part
                starts the same in every setup that has it.
        """
        super().__init__()

        def seeded(name: str, build: Callable[[], nn.Module]) -> nn.Module:
            """Return one part built on its own stream of the seed."""
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(named_seed(seed, name))
                return build()

        # One encoder per instrument, each from its patches to tokens
        self.encoders = nn.ModuleDict(
            {
                name: seeded(
                    f"encoders.{name}",
                    lambda: Encoder(
                        shape,
                        encoder_conditions[name],
                        encoder_dim,
                        encoder_heads,
                        encoder_depth,
                    ),
                )
                for name, shape in shapes.items()
            }
        )
        # One stack the instruments read together pass through, on shared weights
        self.crossencoder = seeded(
            "crossencoder",
            lambda: CrossSensorEncoder(encoder_dim, encoder_heads, crossencoder_depth),
        )
        # One decoder per instrument, each writing its patches back from the tokens
        self.decoders = nn.ModuleDict(
            {
                name: seeded(
                    f"decoders.{name}",
                    lambda: Decoder(
                        shape,
                        decoder_conditions[name],
                        encoder_dim,
                        decoder_dim,
                        decoder_heads,
                        decoder_depth,
                    ),
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
        with_cls = [
            one.readable_and_position_with_cls(readable[name])
            for name, one in batch.items()
        ]
        cls, tokens = self.crossencoder(
            [self.encoders[name](one, readable[name]) for name, one in batch.items()],
            [counted for counted, _ in with_cls],
            [position for _, position in with_cls],
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
        with_cls = [
            batch[name].readable_and_position_with_cls(readable[name]) for name in read
        ]
        return Context(
            torch.cat([cls.unsqueeze(1), *(tokens[name] for name in read)], 1),
            functional.pad(torch.cat([at for _, at in with_cls], 1), (0, 0, 1, 0)),
            functional.pad(
                torch.cat([on for on, _ in with_cls], 1), (1, 0), value=True
            ),  # (B, S)
        )

    def forward(
        self,
        batch: dict[str, Tokens],
        visible: dict[str, Tensor],
        hidden: dict[str, Tensor],
        pooling: str,
    ) -> tuple[dict[str, dict[str, Tensor]], dict[str, Tensor]]:
        """Return every instrument's hidden patches and its tile vector.

        Args:
            batch: Each instrument's patches over the batch.
            visible: Which patches each encoder may read. (B, K)
            hidden: Which patches are predicted, none of them padding. (B, K)
            pooling: What a tile vector is, GAP of the visible tokens or the CLS.

        Returns:
            predictions: Under "umr" each instrument's patches read from its own
                tokens with the CLS of every instrument, under "cmr" from every
                other instrument's without it. (B, K, *P)
            pooled: Each instrument's tile vector as one unit vector, in float.
                (B, D)
        """
        read = {name: one.kept(visible[name]) for name, one in batch.items()}
        seen = {name: kept for name, (kept, _) in read.items()}
        readable = {name: counted for name, (_, counted) in read.items()}
        cls, tokens = self.shared_tokens(seen, readable)
        predictions = {"umr": {}, "cmr": {}}
        for name, one in batch.items():
            others = [other for other in batch if other != name]
            sources = self.context(cls, tokens, seen, readable, others)
            predictions["umr"][name], predictions["cmr"][name] = self.decoders[name](
                self.context(cls, tokens, seen, readable, [name]),
                Context(*(one[:, 1:] for one in sources)),
                one.position,
                hidden[name],
                one.condition,
            )  # (B, K, *P) each
        pooled = {
            name: functional.normalize(tokens[name][:, 0].float(), dim=-1)
            if pooling == CLS
            else global_average_pooling(tokens[name][:, 1:].float(), readable[name])
            for name in batch
        }
        return predictions, pooled
