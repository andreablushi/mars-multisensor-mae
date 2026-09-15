"""The cross-sensor masked autoencoder, assembled from its parts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from torch import Tensor, nn

from architecture.components.crossattention_fusion import (
    CrossAttentionFusion,
    cell_positions,
)
from architecture.components.crossencoder import CrossSensorEncoder
from architecture.components.decoder import Decoder
from architecture.components.encoder import Encoder
from architecture.models import Cells, FeatureGrid, Tokens
from config.schema import ModelConfig


@dataclass(frozen=True, slots=True)
class Reconstruction:
    """What one masked pass hands the loss.

    Attributes:
        predictions: The predicted patches, by the sensor asked and the sensor read.
        grid: The grid every instrument the feature holds was read into.
        grids: The grid each instrument alone was read into, keyed as ODE names it.
    """

    predictions: dict[tuple[str, str], Tensor]
    grid: FeatureGrid
    grids: dict[str, FeatureGrid]


class CrossSensorMAE(nn.Module):
    """An encoder and a decoder per sensor, one shared encoder aligning them between."""

    def __init__(
        self,
        shapes: dict[str, tuple[int, ...]],
        axes: dict[str, tuple[str, ...]],
        strides: dict[str, float],
        config: ModelConfig,
    ) -> None:
        """Build every part for the instruments the model reads."""
        super().__init__()
        # Sensor-specific input projection stems and positional/modality encoders
        self.encoders = nn.ModuleDict(
            {
                name: Encoder(
                    shape,
                    axes[name],
                    config.encoder_dim,
                    config.encoder_heads,
                    config.encoder_depth,
                    strides[name],
                )
                for name, shape in shapes.items()
            }
        )
        # Shared backbone projecting all sensor tokens into a common space
        self.crossencoder = CrossSensorEncoder(
            config.encoder_dim, config.encoder_heads, config.crossencoder_depth
        )
        # The one place the instruments meet, each cell read from what reaches it
        self.fusion = CrossAttentionFusion(
            config.encoder_dim, config.encoder_heads, config.cell_m
        )
        # Sensor-specific reconstruction heads for target patch recovery
        self.decoders = nn.ModuleDict(
            {
                name: Decoder(
                    shape,
                    axes[name],
                    config.encoder_dim,
                    config.decoder_dim,
                    config.decoder_heads,
                    config.decoder_depth,
                    min(strides[name], config.cell_m),
                )
                for name, shape in shapes.items()
            }
        )
        self.dim = config.encoder_dim

    def shared_tokens(self, name: str, tokens: Tokens, visible: Tensor) -> Tensor:
        """Return one instrument's patches in the space every instrument shares."""
        # Guard against empty/absent sensor token inputs
        if tokens.values.shape[1] == 0:
            return tokens.values.new_zeros(
                tokens.values.shape[0], 0, self.dim
            )  # (B, 0, D)
        # Process through sensor-specific stem then map to shared latent space
        encoded = self.encoders[name](
            tokens.values, tokens.channels, tokens.valid, tokens.position, visible
        )  # (B, K, D)
        return self.crossencoder(encoded, visible)  # (B, K, D)

    def encode(self, batch: dict[str, Tokens]) -> dict[str, Tensor]:
        """Return every present patch of every instrument as a token, none hidden."""
        # Extract fully unmasked representations across all present batch sensors
        return {
            name: self.shared_tokens(name, tokens, tokens.present)
            for name, tokens in batch.items()
        }

    def gridded(
        self,
        encoded: dict[str, Tensor],
        batch: dict[str, Tokens],
        counted: dict[str, Tensor],
        cells: Cells,
        read: Sequence[str],
    ) -> FeatureGrid:
        """Return the grid one set of instruments makes of each feature of a batch."""
        return self.fusion(
            encoded,
            {name: one.position for name, one in batch.items()},
            counted,
            cells,
            read,
        )

    def embed(self, batch: dict[str, Tokens], cells: Cells) -> FeatureGrid:
        """Return the grid standing for each feature, over every instrument it holds."""
        counted = {name: one.present for name, one in batch.items()}
        return self.gridded(self.encode(batch), batch, counted, cells, list(batch))

    def forward(self, batch: dict[str, Tokens], cells: Cells) -> Reconstruction:
        """Return every instrument's hidden patches, predicted from every instrument."""
        # Encode visible (unmasked) context tokens for each sensor
        encoded = {
            name: self.shared_tokens(name, tokens, tokens.visible)
            for name, tokens in batch.items()
        }
        counted = {name: one.visible for name, one in batch.items()}
        # The grid of all the instruments, and the grid each of them makes alone
        whole = self.gridded(encoded, batch, counted, cells, list(batch))
        grids = {
            name: self.gridded(encoded, batch, counted, cells, [name]) for name in batch
        }
        placed = cell_positions(cells.offset, self.fusion.cell_m)  # (B, Q, 6)
        predictions = {}
        # Every patch is predicted from the cells one instrument alone was read into,
        # so no instrument ever reads its own patches back out of the grid it asks.
        for asked, tokens in batch.items():
            hidden = tokens.present & ~tokens.visible  # (B, K)
            for read in batch:
                predictions[asked, read] = self.decoders[asked](
                    grids[read].values,
                    placed,
                    grids[read].occupied,
                    tokens.position,
                    tokens.channels,
                    hidden,
                )  # (B, K, *P)
        return Reconstruction(predictions, whole, grids)
