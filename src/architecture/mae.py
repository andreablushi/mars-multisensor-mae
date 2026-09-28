"""The cross-sensor masked autoencoder, assembled from its parts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from torch import Tensor, nn

from architecture.components.crossattention_fusion import CrossAttentionFusion
from architecture.components.crossencoder import CrossSensorEncoder
from architecture.components.decoder import Decoder
from architecture.components.encoder import Encoder
from architecture.grid import Cells, TileGrid
from architecture.tokens import Tokens

# B = batch, K = patches, Q = cells, D = token channels, P = patch dimensions.


@dataclass(frozen=True, slots=True)
class Reconstruction:
    """What one masked pass hands the loss.

    Attributes:
        predictions: The predicted patches, by the sensor asked and the sensor read.
    """

    predictions: dict[tuple[str, str], Tensor]


class CrossSensorMAE(nn.Module):
    """An encoder and a decoder per sensor, one shared encoder aligning them between."""

    def __init__(
        self,
        shapes: dict[str, tuple[int, ...]],
        axes: dict[str, tuple[str, ...]],
        centres_nm: dict[str, tuple[float, ...] | None],
        strides: dict[str, float],
        encoder_dim: int,
        encoder_heads: int,
        encoder_depth: int,
        crossencoder_depth: int,
        decoder_dim: int,
        cell_m: float,
        delay_rows: int,
    ) -> None:
        """Build every part for the instruments the model reads.

        Args:
            shapes: The shape of one patch of each instrument, keyed as ODE names it.
            axes: What each axis of those patches holds, keyed the same way.
            centres_nm: The wavelength of each channel, in nm, or None without one.
            strides: How far apart two neighbouring patch centres sit, in metres.
            encoder_dim: How wide a token is everywhere but the decoders.
            encoder_heads: How many attention heads every encoder and the fusion run.
            encoder_depth: How many blocks each instrument encoder stacks.
            crossencoder_depth: How many blocks the cross-sensor encoder stacks.
            decoder_dim: How wide a gathered cell is in the decoders.
            cell_m: How far a cell of a tile's grid runs along the ground, in metres.
            delay_rows: How many radar delay rows a cell of that grid spans.
        """
        super().__init__()
        self.encoders = nn.ModuleDict(
            {
                name: Encoder(
                    shape,
                    axes[name],
                    centres_nm[name],
                    encoder_dim,
                    encoder_heads,
                    encoder_depth,
                    strides[name],
                )
                for name, shape in shapes.items()
            }
        )
        self.crossencoder = CrossSensorEncoder(
            encoder_dim, encoder_heads, crossencoder_depth
        )
        # The one place the instruments meet, each cell read from what reaches it
        self.fusion = CrossAttentionFusion(encoder_dim, encoder_heads, cell_m)
        self.decoders = nn.ModuleDict(
            {
                name: Decoder(shape, encoder_dim, decoder_dim, cell_m, delay_rows)
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
        counted = {name: one.present for name, one in batch.items()}
        encoded = self.shared_tokens(batch, counted)
        return self.gridded(encoded, batch, counted, cells, list(batch))

    def forward(self, batch: dict[str, Tokens], cells: Cells) -> Reconstruction:
        """Return every instrument's hidden patches, predicted from every instrument.

        Args:
            batch: Each instrument's patches over the batch, some of them hidden.
            cells: The cells the batch's patches reach.

        Returns:
            reconstruction: The predictions made from each instrument's grid.
        """
        counted = {name: one.visible for name, one in batch.items()}
        encoded = self.shared_tokens(batch, counted)
        # The grid each instrument makes alone, which is all a decoder ever reads
        grids = {
            name: self.gridded(encoded, batch, counted, cells, [name]) for name in batch
        }
        predictions = {}
        # Every patch is predicted from the cells one instrument alone was read into,
        # so no instrument ever reads its own patches back out of the grid it asks.
        for asked, tokens in batch.items():
            for read in batch:
                predictions[asked, read] = self.decoders[asked](
                    grids[read].values,
                    cells.offset,
                    grids[read].occupied,
                    tokens.position,
                )  # (B, K, *P)
        return Reconstruction(predictions)
