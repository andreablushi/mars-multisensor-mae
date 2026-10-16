"""Rebuilding one whole tile's hidden patches, from its own tokens and the others'."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from building.common.layout import Axis
from building.metadata.observation import ObservationMetadata
from building.preprocessing.crism.correction.centre_wavelengths import LAYOUT
from common.maths.geodesy import SPHEROID
from common.models.tile import Tile

from architecture.mae import CrossSensorMAE
from architecture.tokens import Tokens, token_batch_padding
from dataset.models.positioning import read_surface_delays
from dataset.patches import patch_arrays, read_tile_patches
from dataset.store import CLIP, STANDARDISED_BY, DatasetBuild
from training.step import drawn_masks


def reconstruct(
    model: CrossSensorMAE,
    build: DatasetBuild,
    rows: Mapping[str, Sequence[ObservationMetadata]],
    axes: Mapping[str, tuple[str, ...]],
    sizes: Mapping[str, Mapping[str, int]],
    pool: Mapping[str, int],
    shapes: Mapping[str, tuple[int, ...]],
    delay: str,
    mask_ratio: float,
    seed: int,
    device: torch.device,
) -> dict[str, dict[str, np.ndarray]]:
    """Return every patch of a tile, a sounder's of one observation, rebuilt.

    Args:
        model: The model, on the device.
        build: The build the tile is read from.
        rows: The tile's index rows of each instrument.
        axes: What each axis of each instrument's values holds.
        sizes: How far a patch of each sensor runs along each axis it is cut on.
        pool: How many ground samples of a patch each instrument averages into one.
        shapes: The shape of one patch of each instrument as the model reads it.
        delay: The instrument whose rows give every surface patch its delay.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        seed: What fixes the masks.
        device: Where the model runs.

    Returns:
        reconstruction: Each instrument holding any patch: its patches as the model
            read them under "values" and "measured", their place on a grid over
            the tile, a sounder's on its observation's, under "cell", the tile's
            north-west and south-east corners on that grid, none for a sounder,
            under "corners", how they were normalised under "normalization",
            what brings them back to raw values as value * scale + offset under
            "offset" and "scale", which were hidden under "hidden", and the
            hidden ones rebuilt from its own tokens under "umr" and from the
            others' under "cmr".
    """
    delays = read_surface_delays(build, rows[delay])
    # The tile's box, as every observation of it stores it
    described = build.read_observation(rows[delay][0].path).described
    tile = Tile(described["band"], described["column"], **described["box"])
    half = np.array(
        [
            SPHEROID.inv(tile.centre_lon, tile.centre_lat, lon, lat)[2]
            for lon, lat in (
                (tile.centre_lon, tile.max_lat),
                (tile.east_lon, tile.centre_lat),
            )
        ]
    )
    sample = {}
    for name in sizes:
        held = list(rows.get(name, ()))
        sounder = Axis.DELAY in axes[name]
        # A sounder's largest radargram alone, drawn on its own grid
        if sounder and held:
            held = [max(held, key=lambda one: math.prod(one.shape))]
        read = [
            read_tile_patches({name: [record]}, build, sizes, pool, delays)[name]
            for record in held
        ]
        read = [observation for observation in read if observation]
        patches = [patch for observation in read for patch in observation]
        cells = np.array([one.cell for one in patches]).reshape(len(patches), -1)
        corners = np.zeros((0, cells.shape[1]), int)
        if not sounder and patches:
            ground = [at for at, holds in enumerate(axes[name]) if holds == Axis.GROUND]
            own = [np.array([one.cell for one in held])[:, ground] for held in read]
            placed = [
                np.array([[-one.north_m, one.east_m] for one in held]) for held in read
            ]
            # The metres a patch steps along each axis, on the largest observation
            largest = max(range(len(read)), key=lambda at: len(read[at]))
            fit = np.linalg.lstsq(
                np.c_[own[largest], np.ones(len(own[largest]))],
                placed[largest],
                rcond=None,
            )[0]
            step = np.diag(fit[:2])
            # Each observation kept on its own grid, shifted to its place on the tile
            cells[:, ground] = np.concatenate(
                [
                    cell + np.round(np.mean(at / step - cell, axis=0)).astype(int)
                    for cell, at in zip(own, placed, strict=True)
                ]
            )
            corners = np.zeros((2, cells.shape[1]), int)
            corners[:, ground] = np.round(np.array([-half, half]) / step).astype(int)
        sample[name] = patch_arrays(patches, shapes[name], axes[name]) | {
            "cell": cells,
            "corners": corners,
        }
    batch, _ = token_batch_padding([(sample, "")])
    batch = {
        name: Tokens(*(one.to(device) for one in tokens))
        for name, tokens in batch.items()
    }
    generator = torch.Generator(device=device).manual_seed(seed)
    visible, hidden = drawn_masks(batch, mask_ratio, generator)
    model.eval()
    with torch.no_grad():
        predictions = model(batch, visible, hidden)
    stats = build.read_stats()
    # How each instrument was normalised, named by the constants the build applies
    normalization = {
        name: "/".join(STANDARDISED_BY)
        + (
            " per band"
            if any(np.ndim(constants[key]) for key in STANDARDISED_BY)
            else ""
        )
        + (f", clipped at ±{CLIP:g}" if name == LAYOUT.instrument else "")
        for name, constants in stats.items()
    }
    return {
        name: arrays
        | {"normalization": np.array(normalization[name])}
        | dict(
            zip(
                ("offset", "scale"),
                (np.asarray(stats[name][key], np.float32) for key in STANDARDISED_BY),
                strict=True,
            )
        )
        | {"hidden": hidden[name][0].cpu().numpy()}
        | {term: predictions[term][name][0].cpu().numpy() for term in predictions}
        for name, arrays in sample.items()
        if len(arrays["values"])
    }
