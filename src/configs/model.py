"""Building the model a config and a build's patch layout settle."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from building.configs import sharad

from architecture.mae import CrossSensorMAE
from configs.schema import Config
from dataset.patches import patch_shapes
from dataset.store import DatasetBuild


def built_model(
    config: Config, build: DatasetBuild
) -> tuple[
    CrossSensorMAE,
    torch.device,
    dict[str, Mapping[str, int]],
    dict[str, tuple[int, ...]],
]:
    """Return the model a build's patch layout settles, on the GPU when there is one.

    Args:
        config: What the run reads and the model it builds.
        build: The build whose instruments settle the patch layout.

    Returns:
        model: The model, on its device.
        device: Where it runs.
        sizes: How far a patch of each sensor runs along each axis it is cut on.
        shapes: The shape of one patch of each instrument as the model reads it.
    """
    sizes = {name: config.dataset.patchsize[name] for name in config.model.instruments}
    shapes = patch_shapes(build.read_observation_metadata(), sizes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossSensorMAE(
        shapes,
        {
            name: math.prod(shape) if name == sharad.LAYOUT.instrument else 0
            for name, shape in shapes.items()
        },
        {name: len(build.acquisition[name]) for name in shapes},
        config.model.encoder_dim,
        config.model.encoder_heads,
        config.model.encoder_depth,
        config.model.crossencoder_depth,
        config.model.decoder_dim,
        config.model.decoder_heads,
        config.model.decoder_depth,
        config.training.stage == "unimodal",
        config.dataset.seed,
    ).to(device)
    return model, device, sizes, shapes
