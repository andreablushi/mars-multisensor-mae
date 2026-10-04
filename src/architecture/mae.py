"""The cross-sensor masked autoencoder, assembled from its parts."""

from __future__ import annotations

from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from architecture.components.crossattention_fusion import CrossAttentionFusion
from architecture.components.decoder import Decoder
from architecture.components.encoder import Encoder
from architecture.components.transformer import Transformer
from architecture.grid import Cells, TileGrid
from architecture.tokens import Tokens

# B = batch, K = patches, Q = cells, D = token channels, P = patch dimensions.


class CrossSensorMAE(nn.Module):
    """An encoder and a decoder per sensor, one shared encoder aligning them between."""

    def __init__(
        self,
        shapes: dict[str, tuple[int, ...]],
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
        # One encoder per instrument, each from its patches to tokens
        self.encoders = nn.ModuleDict(
            {
                name: Encoder(
                    shape,
                    encoder_dim,
                    encoder_heads,
                    encoder_depth,
                    strides[name],
                )
                for name, shape in shapes.items()
            }
        )
        # One stack every instrument's tokens pass through alone, on shared weights
        self.crossencoder = Transformer(encoder_dim, encoder_heads, crossencoder_depth)
        # The one place the instruments meet, each cell read from what reaches it
        self.fusion = CrossAttentionFusion(encoder_dim, encoder_heads, cell_m)
        # One decoder per instrument, each writing its patches back from the cells
        self.decoders = nn.ModuleDict(
            {
                name: Decoder(
                    shape,
                    encoder_dim,
                    decoder_dim,
                    decoder_heads,
                    decoder_depth,
                    min(strides[name], cell_m),
                )
                for name, shape in shapes.items()
            }
        )
        # The token width, which an instrument with no patch is handed zeros at
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
            # The instrument's own encoder turns its readable patches into tokens
            stem = self.encoders[name](
                tokens.values,
                tokens.measured,
                tokens.position,
                counted[name],
            )  # (B, K, D)
            # The shared stack maps those tokens into the space every instrument shares
            encoded[name] = self.crossencoder(stem, counted[name])  # (B, K, D)
        return encoded

    def embed(
        self, batch: dict[str, Tokens], present: dict[str, Tensor], cells: Cells
    ) -> TileGrid:
        """Return the grid standing for each tile, over every instrument it holds.

        Args:
            batch: Each instrument's patches over the batch.
            present: Which slots hold a patch rather than padding. (B, K)
            cells: The cells the batch's patches reach.

        Returns:
            grid: One vector per cell, over every patch, none hidden.
        """
        # Every real patch of every instrument is encoded, nothing hidden
        encoded = self.shared_tokens(batch, present)
        # The grid all instruments make of each tile together
        return self.fusion(encoded, batch, present, cells, list(batch))

    def forward(
        self,
        batch: dict[str, Tokens],
        visible: dict[str, Tensor],
        hidden: dict[str, Tensor],
        cells: Cells,
    ) -> dict[tuple[str, str], Tensor]:
        """Return every instrument's hidden patches, predicted from every instrument.

        Args:
            batch: Each instrument's patches over the batch.
            visible: Which patches each encoder may read. (B, K)
            hidden: Which patches are predicted, none of them padding. (B, K)
            cells: The cells the batch's patches reach.

        Returns:
            predictions: The predicted patches, by the instrument asked and the one
                read. (B, K, *P)
        """
        # Only the visible patches are encoded
        encoded = self.shared_tokens(batch, visible)
        # The grid each instrument makes alone, which is all a decoder ever reads
        grids = {
            name: self.fusion(encoded, batch, visible, cells, [name]) for name in batch
        }
        # Where each cell sits and reaches, which every decoder places cells by
        placed = cells.position  # (B, Q, 6)
        predictions = {}
        # Every patch is predicted from the cells one instrument alone was read into,
        # so no instrument ever reads its own patches back out of the grid it asks.
        for asked, tokens in batch.items():
            for read in batch:
                # Recomputed on the way back, so only one decoder pass is held at once
                predictions[asked, read] = checkpoint(
                    self.decoders[asked],
                    grids[read].values,
                    placed,
                    grids[read].occupied,
                    tokens.position,
                    hidden[asked],
                    use_reentrant=False,
                )  # (B, K, *P)
        return predictions
