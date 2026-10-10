"""Writing back the hidden middle of one observation of each instrument."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from building.common.layout import Axis
from building.metadata.observation import ObservationMetadata

from architecture.mae import CrossSensorMAE
from architecture.tokens import Tokens
from dataset.models.positioning import read_surface_delays
from dataset.patches import patch_arrays, read_tile_patches
from dataset.store import DatasetBuild
from evaluation.results import Reconstruction


def reconstructed_tile(
    model: CrossSensorMAE,
    build: DatasetBuild,
    rows: Mapping[str, list[ObservationMetadata]],
    axes: Mapping[str, tuple[str, ...]],
    sizes: Mapping[str, Mapping[str, int]],
    pool: Mapping[str, int],
    shapes: Mapping[str, tuple[int, ...]],
    delay: str,
    mask_ratio: float,
    device: torch.device,
) -> dict[str, Reconstruction]:
    """Return one tile written back, each instrument's observation hidden in its middle.

    Args:
        model: The model, on the device.
        build: The build the tile is read from.
        rows: The tile's index rows of each sensor, keyed as ODE names it.
        axes: What each axis of each instrument's values holds.
        sizes: How far a patch of each instrument runs along each axis it is cut on.
        pool: How many ground samples of a patch each instrument averages into one.
        shapes: The shape of one patch of each instrument as the model reads it.
        delay: The instrument whose rows give every surface patch its delay.
        mask_ratio: The share of each observation's patches hidden, as whole
            columns in its middle.
        device: Where the model runs.

    Returns:
        tiles: Each instrument's patches of its first observation holding any, what
            was read and what was written.
    """
    delays = read_surface_delays(build, rows[delay])  # (N, 3)
    batch, visible, hidden, cells = {}, {}, {}, {}
    for name in sizes:
        drawn = next(
            (
                patches
                for record in rows.get(name, ())
                if (
                    patches := read_tile_patches(
                        {name: [record]}, build, {name: sizes[name]}, pool, delays
                    )[name]
                )
            ),
            None,
        )
        if drawn is None:
            continue
        arrays = patch_arrays(drawn, shapes[name], axes[name])
        kept = [at for at, holds in enumerate(axes[name]) if holds != Axis.WAVELENGTH]
        cells[name] = np.array([[one.cell[at] for at in kept] for one in drawn])
        columns = cells[name][:, 1] - cells[name][:, 1].min()  # (K,)
        counts = np.bincount(columns)  # (C,)
        centre = (np.cumsum(counts) - counts / 2) / counts.sum()  # (C,)
        shut = ((centre >= (1 - mask_ratio) / 2) & (centre < (1 + mask_ratio) / 2))[
            columns
        ]
        batch[name] = Tokens(
            *(
                torch.as_tensor(arrays[key], device=device)[None]
                for key in ("values", "measured", "position")
            )
        )
        visible[name] = torch.as_tensor(~shut, device=device)[None]
        hidden[name] = torch.as_tensor(shut, device=device)[None]
    model.eval()
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        written = model(batch, visible, hidden)
    return {
        name: Reconstruction(
            values=batch[name].values[0].cpu().numpy(),
            measured=batch[name].measured[0].cpu().numpy(),
            cell=cells[name],
            visible=visible[name][0].cpu().numpy(),
            hidden=hidden[name][0].cpu().numpy(),
            umr=written["umr"][name][0].float().cpu().numpy(),
            cmr=written["cmr"][name][0].float().cpu().numpy(),
        )
        for name in batch
    }
