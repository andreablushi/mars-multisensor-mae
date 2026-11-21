"""Building the model a config and a build settle."""

from __future__ import annotations

import torch

from mars_multisensor_mae.architecture.load import built_cross_sensor_mae
from mars_multisensor_mae.architecture.mae import CrossSensorMAE
from mars_multisensor_mae.configs.schema import Config
from mars_multisensor_mae.dataset.models.patch import PatchLayout
from mars_multisensor_mae.dataset.store import DatasetBuild


def built_pretrained_model(
    config: Config, build: DatasetBuild
) -> tuple[CrossSensorMAE, torch.device, PatchLayout]:
    """Return the model a config and a build's patch layout settle, on its device.

    Args:
        config: What the run reads and the model it builds.
        build: The build whose instruments settle the patch layout.

    Returns:
        model: The model, on its device.
        device: Where it runs.
        layout: How each instrument is cut into patches, and the shape the model
            reads one in.
    """
    return built_cross_sensor_mae(
        build,
        config.model.instruments,
        config.dataset.patchsize,
        config.training.acquisition_in_encoders,
        config.model.encoder_dim,
        config.model.encoder_heads,
        config.model.encoder_depth,
        config.model.crossencoder_depth,
        config.model.decoder_dim,
        config.model.decoder_heads,
        config.model.decoder_depth,
        config.model.projector_dim,
        config.dataset.seed,
    )
