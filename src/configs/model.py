"""Building the model a config and a build's patch layout settle."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from mars_multisensor_dataset.building.configs import sharad
from torch import Tensor

from architecture.mae import CrossSensorMAE
from benchmarking.architecture.model import PROTOCOLS, BenchmarkModel
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
        config.dataset.seed,
    ).to(device)
    return model, device, sizes, shapes


def built_benchmark_model(
    config: Config,
    mae: CrossSensorMAE,
    device: torch.device,
    heads: Mapping[str, Mapping[str, Tensor]] | None = None,
) -> dict[str, BenchmarkModel]:
    """Return each protocol's model over one MAE, its probes drawn alike.

    Args:
        config: What the run reads and the model it benchmarks.
        mae: The MAE, left as it is.
        device: Where the models run.
        heads: The trained weights to load, keyed by protocol, then by parameter,
            or None to keep them as drawn.

    Returns:
        models: The attentive probe over the frozen MAE and over the MAE with its
            encoders trained, on the device, keyed by protocol.
    """
    models = {}
    for protocol, finetuned in PROTOCOLS.items():
        model = BenchmarkModel(
            mae,
            config.model.encoder_dim,
            config.model.encoder_heads,
            config.benchmark.probe_depth,
            finetuned,
            config.dataset.seed,
        ).to(device)
        if heads:
            model.load_state_dict(model.state_dict() | dict(heads[protocol]))
        models[protocol] = model
    return models
