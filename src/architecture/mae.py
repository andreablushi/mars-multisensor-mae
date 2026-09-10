"""The cross-sensor masked autoencoder, assembled from its parts."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from architecture.components.crossencoder import CrossSensorEncoder
from architecture.components.decoder import Decoder
from architecture.components.encoder import Encoder
from architecture.swath import unit_directions
from architecture.tokens import Tokens
from config.schema import ModelConfig


@dataclass(frozen=True, slots=True)
class Reconstruction:
    """What one masked pass hands the loss.

    Attributes:
        predictions: The predicted hidden patches of each instrument, read from
            each instrument's visible tokens, keyed (predicted, read).
            (B, K, *P)
        tokens: Each instrument's tokens on the sphere, meaningful where
            visible, keyed as ODE names it. (B, K, L)
    """

    predictions: dict[tuple[str, str], Tensor]
    tokens: dict[str, Tensor]


class CrossSensorMAE(nn.Module):
    """An encoder and a decoder per instrument, one cross-sensor encoder between.

    Attributes:
        encoders: Each instrument's own encoder, keyed as ODE names it.
        crossencoder: The cross-sensor encoder every instrument's tokens pass
            through, which lands them at the latent width.
        decoders: Each instrument's own decoder.
        latent: The latent width.
        kappa: How tightly a token drawn while training stays about its mean
            direction.
    """

    def __init__(self, shapes: dict[str, tuple[int, ...]], config: ModelConfig) -> None:
        """Build every part for the instruments the model reads.

        Args:
            shapes: The shape of one patch of each instrument, keyed as ODE
                names it.
            config: How wide and deep each part is.
        """
        super().__init__()
        self.encoders = nn.ModuleDict(
            {
                name: Encoder(shape, config.dim, config.heads, config.depth)
                for name, shape in shapes.items()
            }
        )
        self.crossencoder = CrossSensorEncoder(
            config.dim, config.heads, config.crossencoder_depth, config.latent
        )
        self.decoders = nn.ModuleDict(
            {
                name: Decoder(
                    shape,
                    config.latent,
                    config.decoder_dim,
                    config.decoder_heads,
                    config.decoder_depth,
                )
                for name, shape in shapes.items()
            }
        )
        self.latent = config.latent
        self.kappa = config.kappa

    def swath_tokens(self, name: str, tokens: Tokens, visible: Tensor) -> Tensor:
        """Return one instrument's visible patches as points on the sphere.

        Args:
            name: The instrument, as ODE names it.
            tokens: Its patches over the batch.
            visible: Which of them its encoder may read. (B, K)

        Returns:
            directions: One unit vector per slot, meaningful where visible.
                (B, K, L)
        """
        if tokens.values.shape[1] == 0:
            return tokens.values.new_zeros(
                tokens.values.shape[0], 0, self.latent
            )  # (B, 0, L)
        encoded = self.encoders[name](
            tokens.values, tokens.position, visible
        )  # (B, K, D)
        shared = self.crossencoder(encoded, visible)  # (B, K, L)
        return unit_directions(shared, self.kappa, self.training)  # (B, K, L)

    def encode(self, batch: dict[str, Tokens]) -> dict[str, Tensor]:
        """Return every present patch of every instrument on the sphere, none hidden.

        Args:
            batch: Each instrument's patches over the batch, keyed as ODE names
                it.

        Returns:
            tokens: Each instrument's unit vectors, meaningful where present.
                (B, K, L)
        """
        return {
            name: self.swath_tokens(name, tokens, tokens.present)
            for name, tokens in batch.items()
        }

    def forward(self, batch: dict[str, Tokens]) -> Reconstruction:
        """Return every instrument's hidden patches, predicted from every instrument.

        Args:
            batch: Each instrument's patches over the batch, keyed as ODE names
                it, which say which of them are hidden from its encoder.

        Returns:
            reconstruction: The predictions, and the tokens they were read from.
        """
        encoded = {
            name: self.swath_tokens(name, tokens, tokens.visible)
            for name, tokens in batch.items()
        }
        predictions = {}
        for asked, tokens in batch.items():
            hidden = tokens.present & ~tokens.visible  # (B, K)
            for read in batch:
                predictions[asked, read] = self.decoders[asked](
                    encoded[read],
                    batch[read].position,
                    batch[read].visible,
                    tokens.position,
                    hidden,
                )  # (B, K, *P)
        return Reconstruction(predictions, encoded)
