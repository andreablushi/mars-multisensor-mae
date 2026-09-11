"""The cross-sensor masked autoencoder, assembled from its parts."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from architecture.components.crossencoder import CrossSensorEncoder
from architecture.components.decoder import Decoder
from architecture.components.encoder import Encoder
from architecture.fusion import feature_latent
from architecture.tokens import Tokens
from config.schema import ModelConfig


@dataclass(frozen=True, slots=True)
class Reconstruction:
    """What one masked pass hands the loss.

    Attributes:
        predictions: The predicted hidden patches of each instrument, read from
            each instrument's visible tokens, keyed (predicted, read).
            (B, K, *P)
        tokens: Each instrument's tokens in the space they share, meaningful
            where visible, keyed as ODE names it. (B, K, D)
    """

    predictions: dict[tuple[str, str], Tensor]
    tokens: dict[str, Tensor]


class CrossSensorMAE(nn.Module):
    """An encoder and a decoder per instrument, one cross-sensor encoder between.

    Attributes:
        encoders: Each instrument's own encoder, keyed as ODE names it.
        crossencoder: The cross-sensor encoder every instrument's tokens pass
            through.
        decoders: Each instrument's own decoder.
        dim: The token width the cross-sensor encoder hands tokens at.
    """

    def __init__(
        self,
        shapes: dict[str, tuple[int, ...]],
        resolutions: dict[str, float],
        config: ModelConfig,
    ) -> None:
        """Build every part for the instruments the model reads.

        Args:
            shapes: The shape of one patch of each instrument, keyed as ODE
                names it.
            resolutions: How much ground one sample of each instrument spans,
                in metres, which sets the shortest period its positions are
                read at.
            config: How wide and deep each part is.
        """
        super().__init__()
        self.encoders = nn.ModuleDict(
            {
                name: Encoder(
                    shape, config.dim, config.heads, config.depth, resolutions[name]
                )
                for name, shape in shapes.items()
            }
        )
        self.crossencoder = CrossSensorEncoder(
            config.dim, config.heads, config.crossencoder_depth
        )
        self.decoders = nn.ModuleDict(
            {
                name: Decoder(
                    shape,
                    config.dim,
                    config.decoder_dim,
                    config.decoder_heads,
                    config.decoder_depth,
                    resolutions[name],
                )
                for name, shape in shapes.items()
            }
        )
        self.dim = config.dim

    def shared_tokens(self, name: str, tokens: Tokens, visible: Tensor) -> Tensor:
        """Return one instrument's patches in the space every instrument shares.

        Args:
            name: The instrument, as ODE names it.
            tokens: Its patches over the batch.
            visible: Which of them its encoder may read. (B, K)

        Returns:
            tokens: One token per slot, meaningful where visible. (B, K, D)
        """
        if tokens.values.shape[1] == 0:
            return tokens.values.new_zeros(
                tokens.values.shape[0], 0, self.dim
            )  # (B, 0, D)
        encoded = self.encoders[name](
            tokens.values, tokens.position, visible
        )  # (B, K, D)
        return self.crossencoder(encoded, visible)  # (B, K, D)

    def encode(self, batch: dict[str, Tokens]) -> dict[str, Tensor]:
        """Return every present patch of every instrument as a token, none hidden.

        Args:
            batch: Each instrument's patches over the batch, keyed as ODE names
                it.

        Returns:
            tokens: Each instrument's tokens, meaningful where present.
                (B, K, D)
        """
        return {
            name: self.shared_tokens(name, tokens, tokens.present)
            for name, tokens in batch.items()
        }

    def embed(self, batch: dict[str, Tokens]) -> Tensor:
        """Return the one vector standing for each feature of a batch.

        Args:
            batch: Each instrument's patches over the batch, keyed as ODE names
                it.

        Returns:
            latent: One vector per feature, over every patch it holds of every
                instrument, none hidden. (B, D)
        """
        return feature_latent(
            self.encode(batch), {name: one.present for name, one in batch.items()}
        )  # (B, D)

    def forward(self, batch: dict[str, Tokens]) -> Reconstruction:
        """Return every instrument's hidden patches, predicted from every instrument.

        Args:
            batch: Each instrument's patches over the batch, keyed as ODE names
                it, which say which of them are hidden from its encoder.

        Returns:
            reconstruction: The predictions, and the tokens they were read from.
        """
        encoded = {
            name: self.shared_tokens(name, tokens, tokens.visible)
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
