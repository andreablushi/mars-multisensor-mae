"""Writing back what one observation of each instrument never showed the model."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from building.common.layout import Axis
from building.metadata.observation import ObservationMetadata
from sklearn.neighbors import NearestNeighbors

from architecture.mae import CrossSensorMAE
from architecture.tokens import Tokens
from dataset.models.positioning import read_surface_delays
from dataset.patches import patch_arrays, read_tile_patches
from dataset.store import DatasetBuild
from evaluation.results import Reconstruction


def unseen_cells(
    cells: np.ndarray, ground: np.ndarray, delays: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the cells of one observation's patch grid over the tile it never saw.

    Args:
        cells: The row and column of each patch in its observation's grid. (K, 2)
        ground: Each patch's centre, east and north in metres. (K, 2)
        delays: Which delay row the ground sounds at, over the tile. (N, 3)

    Returns:
        unseen: The row and column of every cell over the tile no patch holds. (U, 2)
        position: The centre and delay row of each, as a patch gives them. (U, 3)
    """
    extent = delays[:, [1, 0]]  # (N, 2)
    design = np.column_stack([np.ones(len(cells)), cells])  # (K, 3)
    origin, *steps = np.linalg.lstsq(design, ground, rcond=None)[0]
    steps = np.stack(steps)  # (2, 2)
    reached = np.rint((extent - origin) @ np.linalg.inv(steps)).astype(int)  # (N, 2)
    low, high = reached.min(0), reached.max(0)
    seen = set(map(tuple, cells))
    unseen = np.array(
        [
            (row, column)
            for row in range(low[0], high[0] + 1)
            for column in range(low[1], high[1] + 1)
            if (row, column) not in seen
        ],
        dtype=int,
    ).reshape(-1, 2)
    centres = origin + unseen @ steps  # (U, 2)
    tree = NearestNeighbors().fit(extent)
    spacing = np.median(tree.kneighbors(extent, 2)[0][:, 1])
    apart, nearest = (one[:, 0] for one in tree.kneighbors(centres, 1))
    inside = apart <= max(spacing, np.linalg.norm(steps, axis=1).min() / 2)
    position = np.column_stack([centres, delays[nearest, 2] + 0.5])
    return unseen[inside], position[inside]


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
    """Return one tile written back from one observation of each instrument.

    Args:
        model: The model, on the device.
        build: The build the tile is read from.
        rows: The tile's index rows of each sensor, keyed as ODE names it.
        axes: What each axis of each instrument's values holds.
        sizes: How far a patch of each instrument runs along each axis it is cut on.
        pool: How many ground samples of a patch each instrument averages into one.
        shapes: The shape of one patch of each instrument as the model reads it.
        delay: The instrument whose rows give every surface patch its delay.
        mask_ratio: The share of the sounder's track hidden, in its middle.
        device: Where the model runs.

    Returns:
        tiles: Each instrument's patches, what was read and what was written. An
            imager is asked for the tile it never saw, the sounder for its middle.
    """
    delays = read_surface_delays(build, rows[delay])  # (N, 3)
    read = read_tile_patches(
        {name: rows[name][:1] for name in sizes}, build, sizes, pool, delays
    )
    batch, visible, hidden, cells = {}, {}, {}, {}
    for name, drawn in read.items():
        if not drawn:
            continue
        arrays = patch_arrays(drawn, shapes[name], axes[name])
        values, measured = arrays["values"], arrays["measured"]
        position = arrays["position"]
        kept = [at for at, holds in enumerate(axes[name]) if holds != Axis.WAVELENGTH]
        cells[name] = np.array([[one.cell[at] for at in kept] for one in drawn])
        if Axis.DELAY in axes[name]:
            columns = cells[name][:, kept.index(axes[name].index(Axis.GROUND))]
            columns = columns - columns.min()
            count = columns.max() + 1
            first = int(count * (1 - mask_ratio) / 2)
            shut = (columns >= first) & (columns < first + int(count * mask_ratio))
        else:
            unseen, placed = unseen_cells(cells[name], position[:, :2], delays)
            values = np.concatenate(
                [values, np.zeros((len(unseen), *values.shape[1:]))]
            )
            measured = np.concatenate(
                [measured, np.zeros((len(unseen), *measured.shape[1:]), bool)]
            )
            position = np.concatenate([position, placed])
            cells[name] = np.concatenate([cells[name], unseen])
            shut = np.arange(len(values)) >= len(drawn)
        batch[name] = Tokens(
            torch.as_tensor(values, dtype=torch.float32, device=device)[None],
            torch.as_tensor(measured, device=device)[None],
            torch.as_tensor(position, dtype=torch.float32, device=device)[None],
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
