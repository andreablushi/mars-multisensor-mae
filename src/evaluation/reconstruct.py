"""Writing back the hidden middle of one crop of each instrument's observation."""

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

SIDE = 16


def cropped_patches(cells: np.ndarray, filled: np.ndarray, sounder: bool) -> np.ndarray:
    """Return which patches one crop of an observation keeps.

    Args:
        cells: The row and column of each patch in its observation's grid. (K, 2)
        filled: Whether every sample of each patch is a measurement. (K,)
        sounder: Whether the rows are delay, kept whole, rather than ground.

    Returns:
        kept: For a sounder the middle SIDE columns of its track, for an imager the
            largest square of filled patches up to SIDE on a side. (K,)
    """
    rows, columns = (cells - cells.min(0)).T
    if sounder:
        held = np.unique(columns)
        start = held[max((len(held) - SIDE) // 2, 0)]
        return (columns >= start) & (columns < start + SIDE)
    grid = np.zeros((rows.max() + 1, columns.max() + 1), bool)
    grid[rows[filled], columns[filled]] = True
    size = np.zeros((grid.shape[0] + 1, grid.shape[1] + 1), int)
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            if grid[row, column]:
                size[row + 1, column + 1] = (
                    min(
                        size[row, column + 1],
                        size[row + 1, column],
                        size[row, column],
                        SIDE - 1,
                    )
                    + 1
                )
    bottom, right = np.unravel_index(size.argmax(), size.shape)
    side = size[bottom, right]
    return (
        filled
        & (rows >= bottom - side)
        & (rows < bottom)
        & (columns >= right - side)
        & (columns < right)
    )


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
    """Return one tile written back, a crop of each instrument hidden in its middle.

    Args:
        model: The model, on the device.
        build: The build the tile is read from.
        rows: The tile's index rows of each sensor, keyed as ODE names it.
        axes: What each axis of each instrument's values holds.
        sizes: How far a patch of each instrument runs along each axis it is cut on.
        pool: How many ground samples of a patch each instrument averages into one.
        shapes: The shape of one patch of each instrument as the model reads it.
        delay: The instrument whose rows give every surface patch its delay.
        mask_ratio: The share of each crop's columns hidden, in its middle.
        device: Where the model runs.

    Returns:
        tiles: Each instrument's crop of its first observation holding any patch,
            what was read and what was written.
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
        grid = np.array([[one.cell[at] for at in kept] for one in drawn])  # (K, 2)
        filled = arrays["measured"].reshape(len(drawn), -1).all(axis=1)  # (K,)
        crop = cropped_patches(grid, filled, Axis.DELAY in axes[name])
        cells[name] = grid[crop] - grid[crop].min(0)
        columns = cells[name][:, 1]
        count = columns.max() + 1
        width = round(count * mask_ratio)
        first = (count - width) // 2
        shut = (columns >= first) & (columns < first + width)
        batch[name] = Tokens(
            *(
                torch.as_tensor(arrays[key][crop], device=device)[None]
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
