"""Filling an observation's mosaic with what the model makes of its hidden patches."""

from __future__ import annotations

import numpy as np
import torch
import wandb
from building.common.layout import WAVELENGTH
from building.metadata.observation import ObservationMetadata
from matplotlib import pyplot
from matplotlib.figure import Figure
from wandb.sdk.wandb_run import Run

from architecture.mae import CrossSensorMAE
from architecture.tokens import Tokens
from dataset.models.observation import Observation
from dataset.models.split import DatasetSplit
from dataset.patches import cut_patch, patch_arrays, patch_counts
from training.loss import normalised_patches
from training.masking import random_correspondence


def block_indices(observation: Observation, size: int, across: int) -> list[int]:
    """Return the patches of one square block at the start of an observation.

    Args:
        observation: The observation, read whole.
        size: How far a patch of its instrument runs along an axis it is cut on.
        across: How many patches the block runs along each cut axis.

    Returns:
        indices: Which patches the block holds, counted as `cut_patch` counts them.
    """
    counts = patch_counts(observation.values.shape, observation.axes, size)
    spans = [
        range(1) if holds == WAVELENGTH else range(min(across, count))
        for count, holds in zip(counts, observation.axes, strict=True)
    ]
    grid = np.meshgrid(*[np.array(one) for one in spans], indexing="ij")
    return [
        int(at) for at in np.ravel_multi_index([one.ravel() for one in grid], counts)
    ]


def tiled(patches: list[np.ndarray], origins: list[tuple[int, ...]]) -> np.ndarray:
    """Return patches laid back into the ground they were cut from.

    Args:
        patches: Each patch reduced to the two axes it is looked at along.
        origins: Where each of them starts along those same two axes.

    Returns:
        mosaic: The block they cover, nan where no patch reached. (H, W)
    """
    if not patches:
        return np.full((1, 1), np.nan, np.float32)
    height = max(at[0] + one.shape[0] for one, at in zip(patches, origins, strict=True))
    width = max(at[1] + one.shape[1] for one, at in zip(patches, origins, strict=True))
    least = (min(at[0] for at in origins), min(at[1] for at in origins))
    mosaic = np.full((height - least[0], width - least[1]), np.nan, np.float32)
    for one, at in zip(patches, origins, strict=True):
        mosaic[
            at[0] - least[0] : at[0] - least[0] + one.shape[0],
            at[1] - least[1] : at[1] - least[1] + one.shape[1],
        ] = one
    return mosaic


def looked_at(values: np.ndarray, axes: tuple[str, ...]) -> np.ndarray:
    """Return one patch as the two axes it is looked at along.

    Args:
        values: The patch, as the instrument holds it. (*P)
        axes: What each of its axes holds, in that same order.

    Returns:
        flat: The patch with any wavelength axis averaged away. (H, W)
    """
    at = tuple(one for one, holds in enumerate(axes) if holds == WAVELENGTH)
    return values.mean(axis=at) if at else values


def observation_mosaic(
    model: CrossSensorMAE,
    observation: Observation,
    record: ObservationMetadata,
    name: str,
    size: int,
    shape: tuple[int, ...],
    axes: tuple[str, ...],
    wavelengths: tuple[float, ...],
    statistics: dict[str, float],
    heights: np.ndarray,
    across: int,
    mask_ratio: float,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one block of an observation as measured, as filled, and what was hidden.

    The model is trained against patches centred and scaled by their own
    measured samples, so it never predicts a patch's level or contrast. Each
    predicted patch is put back in the instrument's units by the mean and
    deviation of the true patch, which is what makes the two mosaics comparable.

    Args:
        model: The model, which is switched to evaluation.
        observation: The observation, read whole.
        record: Its index row.
        name: Which instrument cut it, as ODE names it.
        size: How far a patch of it runs along an axis it is cut on.
        shape: The shape of one patch of it as the model reads it.
        axes: What each axis of its values holds.
        wavelengths: What each band is centred on, empty for a non-spectral one.
        statistics: What the instrument's values run to over the training split.
        heights: Where the ground stands over the feature. (N, 3)
        across: How many patches the block runs along each cut axis.
        mask_ratio: The share of the block's patches hidden from the encoder.
        generator: What fixes which of them are hidden.
        device: Where the model runs.

    Returns:
        measured: The block as measured, nan where no patch reached. (H, W)
        filled: The same, every hidden patch as the model rebuilt it. (H, W)
        hidden: Whether each sample of the block was hidden from the encoder. (H, W)
    """
    cut = [
        cut_patch(observation, record, at, size, heights, wavelengths)
        for at in block_indices(observation, size, across)
    ]
    drawn = [one for one in cut if one.valid.any()]
    arrays = patch_arrays(drawn, shape, axes, statistics)
    held = {
        key: torch.as_tensor(value).unsqueeze(0).to(device)
        for key, value in arrays.items()
    }
    present = torch.ones(1, len(drawn), dtype=torch.bool, device=device)  # (1, K)
    tokens = Tokens(**held, visible=present, present=present)
    masked = random_correspondence({name: tokens}, mask_ratio, generator)[name]
    model.eval()
    with torch.no_grad():
        prediction = model({name: masked}).predictions[name, name]  # (1, K, *P)
        _, _, mean, deviation = normalised_patches(tokens.values, tokens.valid)
        rebuilt = (prediction * deviation + mean)[0].cpu().numpy()  # (K, *P)
    was_hidden = (~masked.visible[0]).cpu().numpy()  # (K,)
    ground = tuple(at for at, holds in enumerate(axes) if holds != WAVELENGTH)
    origins = [tuple(one.origin[at] for at in ground) for one in drawn]
    truth = [looked_at(one.values.astype(np.float32), axes) for one in drawn]
    shown = [
        looked_at(rebuilt[at], axes) if was_hidden[at] else truth[at]
        for at in range(len(drawn))
    ]
    blanked = [
        np.full_like(truth[at], float(was_hidden[at])) for at in range(len(drawn))
    ]
    return tiled(truth, origins), tiled(shown, origins), tiled(blanked, origins)


def mosaic_figure(
    measured: np.ndarray, filled: np.ndarray, hidden: np.ndarray, name: str
) -> Figure:
    """Return one instrument's block as measured, as filled, and where it was hidden.

    Args:
        measured: The block as the instrument measured it. (H, W)
        filled: The same, every hidden patch as the model rebuilt it. (H, W)
        hidden: Whether each sample was hidden from the encoder. (H, W)
        name: The instrument, as ODE names it.

    Returns:
        figure: The two mosaics side by side on one scale, and what was hidden.
    """
    low, high = np.nanpercentile(measured, [2, 98])
    figure, panels = pyplot.subplots(1, 3, figsize=(13, 4.4))
    for axes, one, title in zip(
        panels,
        (measured, filled, hidden),
        (f"{name} measured", f"{name} filled", "hidden"),
        strict=True,
    ):
        drawn = axes.imshow(
            one,
            cmap="gray" if one is not hidden else "magma",
            vmin=None if one is hidden else low,
            vmax=None if one is hidden else high,
            interpolation="nearest",
        )
        axes.set_title(title, fontsize=9)
        axes.set_xticks([])
        axes.set_yticks([])
        figure.colorbar(drawn, ax=axes, shrink=0.75)
    figure.tight_layout()
    return figure


def report_mosaics(
    run: Run,
    model: CrossSensorMAE,
    split: DatasetSplit,
    across: int,
    mask_ratio: float,
    seed: int,
    device: torch.device,
) -> None:
    """Log one feature's mosaic, measured and filled, for each instrument that holds it.

    Args:
        run: The tracked run.
        model: The model, loaded from a checkpoint and on the device.
        split: The split the feature and everything it is read with come from.
        across: How many patches the block runs along each cut axis.
        mask_ratio: The share of the block's patches hidden from the encoder.
        seed: What fixes which of them are hidden.
        device: Where the model runs.
    """
    identity = split.identities[0]
    rows = split.features[identity]
    heights = split.build.read_heights(rows[split.elevation])
    generator = torch.Generator(device=device).manual_seed(seed)
    for name in split.sizes:
        held = rows.get(name)
        if not held:
            continue
        drawn = mosaic_figure(
            *observation_mosaic(
                model,
                split.build.read_observation(held[0].path),
                held[0],
                name,
                split.sizes[name],
                split.shapes[name],
                split.axes[name],
                split.wavelengths.get(name, ()),
                split.statistics[name],
                heights,
                across,
                mask_ratio,
                generator,
                device,
            ),
            name,
        )
        run.log({f"mosaic/{name}": wandb.Image(drawn)})
        pyplot.close(drawn)
