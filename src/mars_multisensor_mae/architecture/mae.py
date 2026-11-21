"""The cross-sensor masked autoencoder, assembled from its parts."""

from __future__ import annotations

from torch import Tensor, nn

from mars_multisensor_mae.architecture.components.crossencoder import CrossSensorEncoder
from mars_multisensor_mae.architecture.components.decoder import Decoder
from mars_multisensor_mae.architecture.components.encoder import Encoder
from mars_multisensor_mae.architecture.models.context import (
    Context,
    EncodedBatch,
    Pooling,
)
from mars_multisensor_mae.architecture.models.tokens import Tokens
from mars_multisensor_mae.configs.load import named_rng


class CrossSensorMAE(nn.Module):
    """An encoder and a decoder per sensor, one shared encoder aligning them between.

    Attributes:
        encoders: One encoder per instrument, from its patches to tokens.
        crossencoder: The shared stack, whose CLS of every sensor each decoder reads
            beside its own tokens, so the UMR loss trains it.
        decoders: One decoder per instrument, writing its patches back from the tokens.
    """

    def __init__(
        self,
        shapes: dict[str, tuple[int, ...]],
        conditions: dict[str, int],
        acquired: dict[str, int],
        acquired_in_encoders: bool,
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
            conditions: How many values each instrument's encoder reads its patches
                conditioned on.
            acquired: How many acquisition angles each instrument's decoder reads
                each patch with.
            acquired_in_encoders: Whether each encoder reads them too.
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
                    acquired[name] if acquired_in_encoders else 0,
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
        self, batch: dict[str, Tokens], readable: dict[str, Tensor] | None = None
    ) -> EncodedBatch:
        """Return the CLS of every instrument, and each one's tokens in shared space.

        Args:
            batch: Each instrument's patches over the batch.
            readable: Which patches each encoder may read, every present one when
                None. (B, K)

        Returns:
            encoded: The token read over every instrument together, and each one's
                CLS token, then its tokens, meaningful where readable.
        """
        with_cls = {
            name: one.readable_and_position_with_cls(
                one.present if readable is None else readable[name]
            )
            for name, one in batch.items()
        }
        cls, tokens = self.crossencoder(
            [self.encoders[name](one, *with_cls[name]) for name, one in batch.items()],
            [counted for counted, _ in with_cls.values()],
            [position for _, position in with_cls.values()],
        )
        return EncodedBatch(
            cls,
            {
                name: Context(token, position, counted)
                for (name, (counted, position)), token in zip(
                    with_cls.items(), tokens, strict=True
                )
            },
        )

    def forward(
        self,
        batch: dict[str, Tokens],
        visible: dict[str, Tensor],
        hidden: dict[str, Tensor],
        pooling: Pooling,
    ) -> tuple[dict[str, dict[str, Tensor]], dict[str, Tensor]]:
        """Return every instrument's hidden patches and its tile vector.

        Args:
            batch: Each instrument's patches over the batch.
            visible: Which patches each encoder may read. (B, K)
            hidden: Which patches are predicted, none of them padding. (B, K)
            pooling: What a tile vector is, GAP of the visible tokens or the CLS.

        Returns:
            predictions: Under "umr" each instrument's patches from its own tokens and
                the overall CLS, under "cmr" from every other's tokens alone. (B, K, *P)
            pooled: Each instrument's tile vector as one unit vector, in float.
                (B, D)
        """
        read = {
            name: one.selected_slots_packed_to_front(visible[name])
            for name, one in batch.items()
        }
        seen = {name: packed for name, (packed, _) in read.items()}
        readable = {name: filled for name, (_, filled) in read.items()}
        encoded = self.embed_tokens(seen, readable)
        predictions = {"umr": {}, "cmr": {}}
        for name, one in batch.items():
            # Only the hidden slots are asked, packed to the front, then put back
            asked, filled = one.selected_slots_packed_to_front(hidden[name])
            # UMR reads the overall CLS and its own tokens, CMR every other instrument's
            contexts = {
                "umr": encoded.unimodal_reconstruction_context(name),
                "cmr": encoded.cross_modal_reconstruction_context(name),
            }
            for kind, context in contexts.items():
                written = self.decoders[name](
                    context, asked.position, asked.acquisition, filled
                )  # (B, K', *P)
                placed = written.new_zeros(*hidden[name].shape, *written.shape[2:])
                placed[hidden[name]] = written[filled]
                predictions[kind][name] = placed  # (B, K, *P)
        return predictions, encoded.pooled_tile_vectors(pooling)
