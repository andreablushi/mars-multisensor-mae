"""Building the model a config and a build settle, and loading its logged weights."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from mars_multisensor_dataset.building.configs import sharad
from torch import Tensor
from wandb.sdk.wandb_run import Run

from architecture.mae import CrossSensorMAE
from benchmarking.architecture.model import PROTOCOLS, BenchmarkModel
from configs.schema import Config
from dataset.models.patch import PatchLayout
from dataset.patches import patch_shapes
from dataset.store import DatasetBuild
from logs.tracker import console_logger, logged_checkpoint

log = console_logger(__name__)


def built_pretrained_model(
    config: Config, build: DatasetBuild
) -> tuple[CrossSensorMAE, torch.device, PatchLayout]:
    """Return the model a build's patch layout settles, on the GPU when there is one.

    Args:
        config: What the run reads and the model it builds.
        build: The build whose instruments settle the patch layout.

    Returns:
        model: The model, on its device.
        device: Where it runs.
        layout: How each instrument is cut into patches, and the shape the model
            reads one in.
    """
    sizes = {name: config.dataset.patchsize[name] for name in config.model.instruments}
    shapes = patch_shapes(build.read_observation_metadata(), sizes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
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
        config.model.projector_dim,
        config.dataset.seed,
    ).to(device)
    return model, device, PatchLayout(sizes, shapes)


def trained_model(model: CrossSensorMAE, run: Run, run_name: str) -> CrossSensorMAE:
    """Return the model with a logged run's latest weights, a "random" run's as drawn.

    Args:
        model: The model, as built.
        run: The tracked run that reads the weights.
        run_name: What the trained run is called, which names its model, a last
            part of "random" keeping the seed's own weights.

    Returns:
        model: The same model, its weights loaded.
    """
    if run_name.split("/")[-1] != "random":
        held = torch.load(logged_checkpoint(run, run_name), map_location="cpu")
        model.load_state_dict(held["model"])
        log.info("%s trained for %d steps", run_name, held["step"])
    return model


def built_benchmark_model(
    config: Config,
    mae: CrossSensorMAE,
    device: torch.device,
    heads: Mapping[str, Mapping[str, Tensor]] | None = None,
) -> dict[str, BenchmarkModel]:
    """Return each protocol's model over one MAE, its probes drawn alike.

    Args:
        config: What the run reads and the model it benchmarks.
        mae: The MAE, its weights left as they are, shared by the frozen protocol.
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
            config.benchmark.probe.depth,
            finetuned,
            config.dataset.seed,
        ).to(device)
        if heads:
            model.load_state_dict(model.state_dict() | dict(heads[protocol]))
        models[protocol] = model
    return models
