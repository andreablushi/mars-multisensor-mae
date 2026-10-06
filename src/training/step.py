"""One training step on the model device: patches hidden, then predicted."""

from __future__ import annotations

from collections import defaultdict

import torch
from torch import Tensor

from architecture.mae import CrossSensorMAE
from architecture.tokens import Tokens
from training.loss import cmr_errors, loss_terms, scored_counts, umr_errors


def device_batch(batch: dict[str, Tokens], device: torch.device) -> dict[str, Tokens]:
    """Return one batch held on the model device.

    Args:
        batch: Each instrument's patches over the batch.
        device: Where the model runs.

    Returns:
        batch: The patches, on the device.
    """
    return {
        name: Tokens(*(one.to(device, non_blocking=True) for one in tokens))
        for name, tokens in batch.items()
    }


def drawn_masks(
    batch: dict[str, Tokens],
    mask_ratio: float,
    generator: torch.Generator,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Return which patches are read and which are hidden.

    Args:
        batch: Each instrument's patches over the batch, on the device.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        generator: What fixes the mask, on the device.

    Returns:
        visible: Which patches each encoder reads, drawn for each on its own. (B, K)
        hidden: Which patches are predicted, none of them padding. (B, K)
    """
    visible, hidden = {}, {}
    for name, tokens in batch.items():
        present = tokens.present  # (B, K)
        noise = torch.rand(present.shape, generator=generator, device=present.device)
        # Padding is given the highest noise, so only present patches are hidden.
        order = noise.masked_fill(~present, 2.0).argsort(dim=1)  # (B, K)
        rank = order.argsort(dim=1)  # (B, K)
        drawn = rank < (mask_ratio * present.sum(1, keepdim=True)).floor()  # (B, K)
        visible[name] = present & ~drawn  # (B, K)
        hidden[name] = present & drawn  # (B, K)
    return visible, hidden


def batch_errors(
    model: CrossSensorMAE,
    batch: dict[str, Tokens],
    visible: dict[str, Tensor],
    hidden: dict[str, Tensor],
) -> dict[tuple[str, str], Tensor]:
    """Return each term's error over one batch, summed over the patches it scores.

    Args:
        model: The model, on the device.
        batch: Each instrument's patches over the batch, on the device.
        visible: Which patches each encoder reads. (B, K)
        hidden: Which patches are predicted. (B, K)

    Returns:
        sums: Per term and instrument, the summed error. ()
    """
    device = next(iter(visible.values())).device
    with torch.autocast(device.type, dtype=torch.bfloat16):
        predictions = model(batch, visible, hidden)
    errors = {
        "umr": umr_errors(predictions["umr"], batch, hidden),
        "cmr": cmr_errors(predictions["cmr"], batch, visible, hidden),
    }
    return {
        (term, name): summed
        for term, held in errors.items()
        for name, summed in held.items()
    }


def step_terms(
    model: CrossSensorMAE,
    passes: list[tuple[dict[str, Tokens], list[str]]],
    mask_ratio: float,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, Tensor]:
    """Return one step's loss terms, its gradients added up over every pass.

    Args:
        model: The model, on the device.
        passes: The batches the step is split into, each as the loader hands it.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        generator: What fixes the masks, on the device.
        device: Where the model runs.

    Returns:
        terms: Every loss term over the whole step, as if it were one batch.
    """
    # Every pass is drawn first, so each term is averaged over the whole step
    drawn, counts = [], defaultdict(float)
    for batch, _ in passes:
        batch = device_batch(batch, device)
        visible, hidden = drawn_masks(batch, mask_ratio, generator)
        drawn.append((visible, hidden))
        for term, count in scored_counts(visible, hidden).items():
            counts[term] += count
    sums = defaultdict(float)
    # One pass on the device at a time, its gradients added to the step's
    for (batch, _), masks in zip(passes, drawn, strict=True):
        batch = device_batch(batch, device)
        held = batch_errors(model, batch, *masks)
        loss_terms(held, counts)["loss"].backward()
        for term, value in held.items():
            sums[term] += value.detach()
    return loss_terms(sums, counts)
