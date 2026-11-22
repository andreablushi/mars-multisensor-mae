"""Building the model a build's patch layout settles, and loading its logged weights."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from mars_multisensor_dataset.building.configs import sharad
from wandb.sdk.wandb_run import Run

from mars_multisensor_mae.architecture.mae import CrossSensorMAE
from mars_multisensor_mae.dataset.models.patch import PatchLayout
from mars_multisensor_mae.dataset.store import DatasetBuild
from mars_multisensor_mae.logs.tracker import console_logger, logged_checkpoint

log = console_logger(__name__)


def built_cross_sensor_mae(
    build: DatasetBuild,
    instruments: Sequence[str],
    patchsize: Mapping[str, Mapping[str, int]],
    acquisition_in_encoders: bool,
    encoder_dim: int,
    encoder_heads: int,
    encoder_depth: int,
    crossencoder_depth: int,
    decoder_dim: int,
    decoder_heads: int,
    decoder_depth: int,
    seed: int,
) -> tuple[CrossSensorMAE, torch.device, PatchLayout]:
    """Return the model a build's patch layout settles, on the GPU when there is one.

    Args:
        build: The build whose instruments settle the patch layout.
        instruments: The instruments whose patches become tokens.
        patchsize: A patch's extent per axis, keyed by instrument.
        acquisition_in_encoders: Whether the encoders read the acquisition angles.
        encoder_dim: How wide a token is everywhere but the decoders.
        encoder_heads: How many attention heads every encoder runs.
        encoder_depth: How many blocks each instrument encoder stacks.
        crossencoder_depth: How many blocks the cross-sensor encoder stacks.
        decoder_dim: How wide a token is in the decoders.
        decoder_heads: How many attention heads the decoders run.
        decoder_depth: How many blocks each decoder stacks.
        seed: What fixes every part's initial weights.

    Returns:
        model: The model, on its device.
        device: Where it runs.
        layout: How each instrument is cut into patches, and the shape the model
            reads one in.
    """
    layout = PatchLayout.from_index(
        build.read_observation_metadata(),
        {name: patchsize[name] for name in instruments},
    )
    shapes = layout.shapes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    model = CrossSensorMAE(
        shapes,
        {
            name: math.prod(shape) if name == sharad.LAYOUT.instrument else 0
            for name, shape in shapes.items()
        },
        {name: len(build.acquisition[name]) for name in shapes},
        acquisition_in_encoders,
        encoder_dim,
        encoder_heads,
        encoder_depth,
        crossencoder_depth,
        decoder_dim,
        decoder_heads,
        decoder_depth,
        seed,
    ).to(device)
    return model, device, layout


def cross_sensor_mae_with_logged_weights(
    model: CrossSensorMAE, run: Run, run_name: str, artifacts_root: Path
) -> CrossSensorMAE:
    """Return the model with a logged run's latest weights, a "random" run's as drawn.

    Args:
        model: The model, as built.
        run: The tracked run that reads the weights.
        run_name: What the trained run is called, which names its model, a last
            part of "random" keeping the seed's own weights.
        artifacts_root: Where downloaded artifacts are kept on this machine.

    Returns:
        model: The same model, its weights loaded.
    """
    if run_name.split("/")[-1] != "random":
        path = logged_checkpoint(run, run_name, artifacts_root)
        held = torch.load(path, map_location="cpu")
        model.load_state_dict(held["model"])
        log.info("%s trained for %d steps", run_name, held["step"])
    return model
