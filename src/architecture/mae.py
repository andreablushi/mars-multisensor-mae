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
from architecture.models import Cells, TileGrid, Tokens


@dataclass(frozen=True, slots=True)
class Reconstruction:
    """What one masked pass hands the loss.

    Attributes:
        predictions: The predicted patches, by the sensor asked and the sensor read.
        grids: The grid each instrument alone was read into, keyed as ODE names it.
    """

    predictions: dict[tuple[str, str], Tensor]
    grids: dict[str, TileGrid]


class CrossSensorMAE(nn.Module):
    """An encoder and a decoder per sensor, one shared encoder aligning them between."""

    def __init__(
        self,
        shapes: dict[str, tuple[int, ...]],
        axes: dict[str, tuple[str, ...]],
        strides: dict[str, float],
        encoder_dim: int,
        encoder_heads: int,
        encoder_depth: int,
        crossencoder_depth: int,
        decoder_dim: int,
        decoder_heads: int,
        decoder_depth: int,
        cell_m: float,
    ) -> None:
        """Build every part for the instruments the model reads.

        Args:
            shapes: The shape of one patch of each instrument, keyed as ODE names it.
            axes: What each axis of those patches holds, keyed the same way.
            strides: How far apart two neighbouring patch centres sit, in metres.
            encoder_dim: How wide a token is everywhere but the decoders.
            encoder_heads: How many attention heads every encoder and the fusion run.
            encoder_depth: How many blocks each instrument encoder stacks.
            crossencoder_depth: How many blocks the cross-sensor encoder stacks.
            decoder_dim: How wide a token is in the decoders.
            decoder_heads: How many attention heads the decoders run.
            decoder_depth: How many blocks each decoder stacks.
            cell_m: How far a cell of a tile's grid runs along the ground, in metres.
        """
        super().__init__()
        # Sensor-specific input projection stems and positional/modality encoders
        self.encoders = nn.ModuleDict(
            {
                name: Encoder(
                    shape,
                    axes[name],
                    encoder_dim,
                    encoder_heads,
                    encoder_depth,
                    strides[name],
                )
                for name, shape in shapes.items()
            }
        )
        # Shared backbone projecting all sensor tokens into a common space
        self.crossencoder = CrossSensorEncoder(
            encoder_dim, encoder_heads, crossencoder_depth
        )
        # The one place the instruments meet, each cell read from what reaches it
        self.fusion = CrossAttentionFusion(encoder_dim, encoder_heads, cell_m)
        # Sensor-specific reconstruction heads for target patch recovery
        self.decoders = nn.ModuleDict(
            {
                name: Decoder(
                    shape,
                    axes[name],
                    encoder_dim,
                    decoder_dim,
                    decoder_heads,
                    decoder_depth,
                    min(strides[name], cell_m),
                )
                for name, shape in shapes.items()
            }
        )
        self.dim = encoder_dim

    def shared_tokens(
        self, batch: dict[str, Tokens], counted: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """Return each instrument's patches in the space every instrument shares.

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
            # Process through sensor-specific stem then map to shared latent space
            stem = self.encoders[name](
                tokens.values,
                tokens.valid,
                tokens.position,
                counted[name],
            )  # (B, K, D)
            encoded[name] = self.crossencoder(stem, counted[name])  # (B, K, D)
        return encoded

    def gridded(
        self,
        encoded: dict[str, Tensor],
        batch: dict[str, Tokens],
        counted: dict[str, Tensor],
        cells: Cells,
        read: Sequence[str],
    ) -> TileGrid:
        """Return the grid one set of instruments makes of each tile of a batch.

        Args:
            encoded: Each instrument's shared tokens. (B, K, D)
            batch: Each instrument's patches over the batch.
            counted: Which of its tokens count: visible while training, else present.
            cells: The cells the batch's patches reach.
            read: Which instruments the grid is built from.

        Returns:
            grid: One vector per cell, of unit length where an instrument reaches it.
        """
        return self.fusion(
            encoded,
            {name: one.position for name, one in batch.items()},
            counted,
            cells,
            read,
        )

    def embed(self, batch: dict[str, Tokens], cells: Cells) -> TileGrid:
        """Return the grid standing for each tile, over every instrument it holds.

        Args:
            batch: Each instrument's patches over the batch.
            cells: The cells the batch's patches reach.

        Returns:
            grid: One vector per cell, over every present patch, none hidden.
        """
        # Extract fully unmasked representations across all present batch sensors
        counted = {name: one.present for name, one in batch.items()}
        encoded = self.shared_tokens(batch, counted)
        return self.gridded(encoded, batch, counted, cells, list(batch))

    def forward(self, batch: dict[str, Tokens], cells: Cells) -> Reconstruction:
        """Return every instrument's hidden patches, predicted from every instrument.

        Args:
            batch: Each instrument's patches over the batch, some of them hidden.
            cells: The cells the batch's patches reach.

        Returns:
            reconstruction: The predictions and the grids they were read from.
        """
        # Encode visible (unmasked) context tokens for each sensor
        counted = {name: one.visible for name, one in batch.items()}
        encoded = self.shared_tokens(batch, counted)
        # The grid each instrument makes alone, which is all a decoder ever reads
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
                    hidden,
                )  # (B, K, *P)
        return Reconstruction(predictions, grids)
