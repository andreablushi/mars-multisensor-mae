"""The cross-sensor masked autoencoder, assembled from its parts."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional

from architecture.components.crossencoder import CrossSensorEncoder
from architecture.components.decoder import Decoder
from architecture.components.encoder import Encoder
from architecture.models.context import Context
from architecture.models.tokens import Tokens
from configs.load import named_rng


def pooled_tokens(
    contexts: dict[str, Context], readable: dict[str, Tensor], pooling: str
) -> dict[str, Tensor]:
    """Return each instrument's tile vector, its CLS or the GAP of its readable tokens.

    Args:
        contexts: Each instrument's CLS token, then its tokens. (B, 1 + K, D)
        readable: Which of each instrument's tokens hold something. (B, K)
        pooling: What a tile vector is, "CLS" or "GAP".

    Returns:
        pooled: Each instrument's tile vector as one unit vector, in float. (B, D)
    """
    return {
        name: functional.normalize(
            held.tokens[:, 0].float()
            if pooling == "CLS"
            else (held.tokens[:, 1:].float() * readable[name].unsqueeze(-1)).sum(1),
            dim=-1,
        )
        for name, held in contexts.items()
    }


class CrossSensorMAE(nn.Module):
    """An encoder and a decoder per sensor, one shared encoder aligning them between.

    The shared encoder also hands back one CLS of every sensor, which every decoder
    reads beside its own tokens unless unimodal, so the UMR loss trains it.
    """

    def __init__(
        self,
        shapes: dict[str, tuple[int, ...]],
        conditions: dict[str, int],
        acquired: dict[str, int],
        encoder_dim: int,
        encoder_heads: int,
        encoder_depth: int,
        crossencoder_depth: int,
        decoder_dim: int,
        decoder_heads: int,
        decoder_depth: int,
        unimodal: bool,
        seed: int,
    ) -> None:
        """Build every part for the instruments the model reads, each from its own seed.

        Args:
            shapes: The shape of one patch of each instrument, keyed as ODE names it.
            conditions: How many values each instrument's encoder reads its patches
                conditioned on.
            acquired: How many acquisition angles each instrument's encoder and
                decoder read each patch with.
            encoder_dim: How wide a token is everywhere but the decoders.
            encoder_heads: How many attention heads every encoder runs.
            encoder_depth: How many blocks each instrument encoder stacks.
            crossencoder_depth: How many blocks the cross-sensor encoder stacks.
            decoder_dim: How wide a token is in the decoders.
            decoder_heads: How many attention heads the decoders run.
            decoder_depth: How many blocks each decoder stacks.
            unimodal: Whether each decoder reads its own tokens alone, with neither
                the CLS of every instrument nor the others' tokens.
            seed: What every part's starting weights are drawn from, so a part
                starts the same in every setup that has it.
        """
        super().__init__()
        self.unimodal = unimodal
        # One encoder per instrument, each from its patches to tokens
        self.encoders = nn.ModuleDict()
        # One stack the instruments read together pass through, on shared weights
        with named_rng(seed, "crossencoder"):
            self.crossencoder = CrossSensorEncoder(
                encoder_dim, encoder_heads, crossencoder_depth
            )
        # One decoder per instrument, each writing its patches back from the tokens
        self.decoders = nn.ModuleDict()
        for name, shape in shapes.items():
            with named_rng(seed, f"encoders.{name}"):
                self.encoders[name] = Encoder(
                    shape,
                    conditions[name],
                    acquired[name],
                    encoder_dim,
                    encoder_heads,
                    encoder_depth,
                )
            with named_rng(seed, f"decoders.{name}"):
                self.decoders[name] = Decoder(
                    shape,
                    acquired[name],
                    encoder_dim,
                    decoder_dim,
                    decoder_heads,
                    decoder_depth,
                )

    def embed_tokens(
        self, batch: dict[str, Tokens], readable: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, Context]]:
        """Return the CLS of every instrument, and each one's tokens in shared space.

        Args:
            batch: Each instrument's patches over the batch.
            readable: Which patches each encoder may read. (B, K)

        Returns:
            cls: The token read over every instrument together. (B, D)
            contexts: Each instrument's CLS token, then its tokens, meaningful where
                readable, with their centres and which of them count. (B, 1 + K, D)
        """
        with_cls = {
            name: one.readable_and_position_with_cls(readable[name])
            for name, one in batch.items()
        }
        cls, tokens = self.crossencoder(
            [self.encoders[name](one, readable[name]) for name, one in batch.items()],
            [counted for counted, _ in with_cls.values()],
            [position for _, position in with_cls.values()],
        )
        return cls, {
            name: Context(token, position, counted)
            for (name, (counted, position)), token in zip(
                with_cls.items(), tokens, strict=True
            )
        }

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
                other instrument's without it; unimodal, "umr" from its own tokens
                alone and "cmr" empty. (B, K, *P)
            pooled: Each instrument's tile vector as one unit vector, in float.
                (B, D)
        """
        read = {name: one.kept(visible[name]) for name, one in batch.items()}
        seen = {name: kept for name, (kept, _) in read.items()}
        readable = {name: counted for name, (_, counted) in read.items()}
        cls, contexts = self.embed_tokens(seen, readable)
        lead = Context(
            cls.unsqueeze(1),
            cls.new_zeros(len(cls), 1, 3),
            cls.new_ones(len(cls), 1, dtype=torch.bool),
        )
        predictions = {"umr": {}, "cmr": {}}
        for name, one in batch.items():
            read = (
                [contexts[name]]
                if self.unimodal
                else [
                    Context.joined([lead, contexts[name]]),
                    Context.joined(
                        [contexts[other] for other in batch if other != name]
                    ),
                ]
            )
            written = self.decoders[name](
                read, one.position, one.acquisition, hidden[name]
            )  # (B, K, *P) each
            for term, prediction in zip(predictions, written, strict=False):
                predictions[term][name] = prediction
        return predictions, pooled_tokens(contexts, readable, pooling)
