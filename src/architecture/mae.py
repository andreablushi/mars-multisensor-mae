"""The cross-sensor masked autoencoder, assembled from its parts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from architecture.components.decoder import Decoder
from architecture.components.encoder import Encoder
from architecture.components.fusion import CrossSensorEncoder
from architecture.distribution.sphere import Sphere
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
        visible: Which of each instrument's patches its encoder read. (B, K)
    """

    predictions: dict[tuple[str, str], Tensor]
    tokens: dict[str, Tensor]
    visible: dict[str, Tensor]


class CrossSensorMAE(nn.Module):
    """An encoder and a decoder per instrument, one cross-sensor encoder between.

    Attributes:
        encoders: Each instrument's own encoder, keyed as ODE names it.
        fusion: The cross-sensor encoder every instrument's tokens pass through.
        sphere: Where every token lands.
        decoders: Each instrument's own decoder.
        mask_ratio: The share of each instrument's patches hidden from its
            encoder.
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
        self.fusion = CrossSensorEncoder(config.dim, config.heads, config.fusion_depth)
        self.sphere = Sphere(config.dim, config.latent, config.kappa)
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
        self.mask_ratio = config.mask_ratio

    def sphere_tokens(self, name: str, tokens: Tokens, visible: Tensor) -> Tensor:
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
            latent = self.sphere.project.out_features
            return tokens.values.new_zeros(
                tokens.values.shape[0], 0, latent
            )  # (B, 0, L)
        encoded = self.encoders[name](
            tokens.values, tokens.position, visible
        )  # (B, K, D)
        shared = self.fusion(encoded, visible)  # (B, K, D)
        return self.sphere(shared)  # (B, K, L)

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
            name: self.sphere_tokens(name, tokens, tokens.present)
            for name, tokens in batch.items()
        }

    def forward(self, batch: dict[str, Tokens]) -> Reconstruction:
        """Return every instrument's hidden patches, predicted from every instrument.

        Args:
            batch: Each instrument's patches over the batch, keyed as ODE names
                it.

        Returns:
            reconstruction: The predictions, the tokens they were read from, and
                what was visible.
        """
        visible = {
            name: tokens.present & ~draw_hidden(tokens.present, self.mask_ratio)
            for name, tokens in batch.items()
        }
        encoded = {
            name: self.sphere_tokens(name, tokens, visible[name])
            for name, tokens in batch.items()
        }
        predictions = {}
        for asked, tokens in batch.items():
            hidden = tokens.present & ~visible[asked]  # (B, K)
            for read in batch:
                predictions[asked, read] = self.decoders[asked](
                    encoded[read],
                    batch[read].position,
                    visible[read],
                    tokens.position,
                    hidden,
                )  # (B, K, *P)
        return Reconstruction(predictions, encoded, visible)


def draw_hidden(present: Tensor, ratio: float) -> Tensor:
    """Return which present patches to hide, a fixed share of each feature's own.

    Args:
        present: Which slots hold a patch. (B, K)
        ratio: The share of each feature's patches to hide.

    Returns:
        hidden: The patches drawn, at random, that many per feature. (B, K)
    """
    noise = torch.rand(present.shape, device=present.device)  # (B, K)
    noise[~present] = torch.inf
    rank = noise.argsort(dim=1).argsort(dim=1)  # (B, K)
    count = (present.sum(dim=1) * ratio).floor()  # (B,)
    return rank < count.unsqueeze(1)  # (B, K)
