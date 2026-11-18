"""One step on the model device: patches hidden, then predicted."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial

import torch
from torch import Tensor

from architecture.mae import CrossSensorMAE
from architecture.models.tokens import Tokens
from training.loss import mim, patch_discrimination, reconstruction_error, total


def autocast_dtype(device: torch.device) -> torch.dtype:
    """Return the half precision autocast runs in on a device.

    Args:
        device: Where the model runs.

    Returns:
        dtype: bfloat16 on CPU and GPUs of capability 8.0 or above, float16 below.
    """
    native = device.type != "cuda" or torch.cuda.get_device_capability(device) >= (8, 0)
    return torch.bfloat16 if native else torch.float16


def masking_random_correspondence(
    batch: dict[str, Tokens],
    mask_ratio: float,
    generators: Mapping[str, torch.Generator],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Return which patches are read and which hidden, each instrument drawn alone.

    Args:
        batch: Each instrument's patches over the batch, on the device.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        generators: What fixes each instrument's mask, on the device.

    Returns:
        visible: Which patches each encoder reads, drawn for each on its own. (B, K)
        hidden: Which patches are predicted, none of them padding. (B, K)
    """
    visible, hidden = {}, {}
    for name, tokens in batch.items():
        present = tokens.present  # (B, K)
        noise = torch.rand(
            present.shape, generator=generators[name], device=present.device
        )
        # Padding is given the highest noise, so only present patches are hidden.
        order = noise.masked_fill(~present, 2.0).argsort(dim=1)  # (B, K)
        visible[name], hidden[name] = visible_and_hidden_from_order(
            order, present, mask_ratio
        )
    return visible, hidden


def masking_identical_correspondence(
    batch: dict[str, Tokens],
    mask_ratio: float,
    generators: Mapping[str, torch.Generator],
    cell_m: float,
    independent: Sequence[str],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Return which patches are read and which hidden, ground cells hidden whole.

    Args:
        batch: Each instrument's patches over the batch, on the device.
        mask_ratio: The share of each tile's ground cells hidden, and of each
            independent instrument's patches.
        generators: What fixes the ground's and each instrument's draws, on the device.
        cell_m: The side of one ground cell, in metres.
        independent: The instruments drawn patch by patch, as random correspondence
            draws them.

    Returns:
        visible: Which patches each encoder reads. (B, K)
        hidden: Which patches are predicted, none of them padding. (B, K)
    """
    visible, hidden = masking_random_correspondence(
        {name: batch[name] for name in independent if name in batch},
        mask_ratio,
        generators,
    )
    shared = {name: one for name, one in batch.items() if name not in independent}
    cells = {
        name: (one.position[..., :2] / cell_m).floor().long()
        for name, one in shared.items()
    }
    every = torch.cat([one.flatten(0, 1) for one in cells.values()])
    low = every.amin(0)
    width = every.amax(0) - low + 1
    index = {
        name: (one[..., 0] - low[0]) * width[1] + one[..., 1] - low[1]
        for name, one in cells.items()
    }
    covered = torch.zeros(
        (next(iter(cells.values())).shape[0], int(width.prod())),
        dtype=torch.int,
        device=low.device,
    )
    for name, at in index.items():
        covered.scatter_add_(1, at, shared[name].present.int())
    field = torch.rand(covered.shape, generator=generators["ground"], device=low.device)
    reached = covered > 0
    order = field.masked_fill(~reached, 2.0).argsort(dim=1)
    _, dropped = visible_and_hidden_from_order(order, reached, mask_ratio)
    for name, tokens in shared.items():
        drawn = dropped.gather(1, index[name])
        visible[name] = tokens.present & ~drawn
        hidden[name] = tokens.present & drawn
    return visible, hidden


def visible_and_hidden_from_order(
    order: Tensor, present: Tensor, mask_ratio: float
) -> tuple[Tensor, Tensor]:
    """Return which patches are read and which hidden, the first of an order hidden.

    Args:
        order: Each tile's slots in the order they are hidden, padding last. (B, K)
        present: Which slots hold a patch rather than padding. (B, K)
        mask_ratio: The share of each tile's patches hidden.

    Returns:
        visible: Which patches are read. (B, K)
        hidden: Which patches are hidden. (B, K)
    """
    rank = order.argsort(dim=1)  # (B, K)
    drawn = rank < (mask_ratio * present.sum(1, keepdim=True)).floor()  # (B, K)
    return present & ~drawn, present & drawn


def selected_masking(
    strategy: str, cell_m: float, independent: Sequence[str]
) -> Callable[..., tuple[dict[str, Tensor], dict[str, Tensor]]]:
    """Return the masking_* strategy a run hides patches by, its settings bound.

    Args:
        strategy: The strategy, as CSMAE names its correspondence.
        cell_m: The side of one ground cell identical correspondence hides whole.
        independent: The instruments identical correspondence draws patch by patch.

    Returns:
        masking: The strategy, called with the batch, the ratio and the generators.
    """
    return {
        "random_correspondence": masking_random_correspondence,
        "identical_correspondence": partial(
            masking_identical_correspondence, cell_m=cell_m, independent=independent
        ),
    }[strategy]


def batch_step(
    model: CrossSensorMAE,
    batch: dict[str, Tokens],
    masking: Callable[..., tuple[dict[str, Tensor], dict[str, Tensor]]],
    mask_ratio: float,
    aligned: Sequence[str],
    pooling: str,
    temperature: float,
    latent_temperature: float,
    measure: str,
    generators: Mapping[str, torch.Generator],
    device: torch.device,
) -> dict[str, Tensor]:
    """Return one batch's loss terms, its patches hidden and predicted.

    Args:
        model: The model, on the device.
        batch: Each instrument's patches over the batch, as the loader hands it.
        masking: The masking_* strategy drawing which patches are read and hidden.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        aligned: The instruments MIM aligns, none for no MIM.
        pooling: What MIM aligns of each instrument, its GAP or its CLS.
        temperature: What MIM divides every cosine by.
        latent_temperature: What latent UMR and CMR divide every cosine by.
        measure: The reconstruction error, MSE or RMSE.
        generators: What fixes each instrument's masks, on the device.
        device: Where the model runs.

    Returns:
        terms: Every loss term over the batch, and the loss.
    """
    batch = {name: tokens.to(device) for name, tokens in batch.items()}
    visible, hidden = masking(batch, mask_ratio, generators)
    with torch.autocast(device.type, dtype=autocast_dtype(device)):
        predictions, pooled = model(batch, visible, hidden, pooling)
    targets = model.target_latents(batch)
    terms = {
        f"{kind}/{name}": patch_discrimination(
            predictions[kind][name], targets[name], hidden[name], latent_temperature
        )
        for kind in ("latent_umr", "latent_cmr")
        for name in targets
    } | {
        f"{kind}/{name}": reconstruction_error(
            predictions[kind][name], batch[name], hidden[name], measure
        )
        for kind in ("umr", "cmr")
        for name in batch
    }
    return total(terms, mim(pooled, aligned, temperature))
