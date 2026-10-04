"""One training step on the model device: patches hidden, instruments left out."""

from __future__ import annotations

from collections import defaultdict

import torch
from torch import Tensor

from architecture.grid import Cells
from architecture.mae import CrossSensorMAE
from architecture.tokens import Tokens
from training.loss import loss_terms, scored_patches, summed_errors


def device_batch(
    batch: dict[str, Tokens], cells: Cells, device: torch.device
) -> tuple[dict[str, Tokens], Cells]:
    """Return one batch held on the model device.

    Args:
        batch: Each instrument's patches over the batch.
        cells: The cells the batch's patches reach.
        device: Where the model runs.

    Returns:
        batch: The patches, on the device.
        cells: The cells, on the device.
    """
    return (
        {
            name: Tokens(*(one.to(device, non_blocking=True) for one in tokens))
            for name, tokens in batch.items()
        },
        Cells(*(one.to(device, non_blocking=True) for one in cells)),
    )


def drawn_masks(
    batch: dict[str, Tokens],
    mask_ratio: float,
    drop_ratio: float,
    generator: torch.Generator,
) -> tuple[
    dict[str, Tensor], dict[str, Tensor], list[str], dict[tuple[str, str], Tensor]
]:
    """Return which patches are read and hidden, which instruments kept, and scored.

    Args:
        batch: Each instrument's patches over the batch, on the device.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        drop_ratio: The chance each instrument is left out of the grid.
        generator: What fixes the mask, on the device.

    Returns:
        visible: Which patches each encoder reads, drawn for each on its own. (B, K)
        hidden: Which patches are predicted, none of them padding. (B, K)
        kept: Which instruments the grid is read from, never none.
        scored: The hidden patches each loss term scores. (B, K)
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
    draw = torch.rand(len(batch), generator=generator, device=generator.device)
    dropped = (draw < drop_ratio).tolist()
    # One instrument is always kept, so the grid is never empty
    if all(dropped):
        dropped[int(draw.argmax())] = False
    kept = [name for name, out in zip(batch, dropped, strict=True) if not out]
    return visible, hidden, kept, scored_patches(batch, visible, hidden, kept)


def batch_errors(
    model: CrossSensorMAE,
    batch: dict[str, Tokens],
    cells: Cells,
    visible: dict[str, Tensor],
    hidden: dict[str, Tensor],
    kept: list[str],
    scored: dict[tuple[str, str], Tensor],
) -> dict[tuple[str, str], Tensor]:
    """Return each term's error over one batch, summed over the patches it scores.

    Args:
        model: The model, on the device.
        batch: Each instrument's patches over the batch, on the device.
        cells: The cells the batch's patches reach, on the device.
        visible: Which patches each encoder reads. (B, K)
        hidden: Which patches are predicted. (B, K)
        kept: Which instruments the grid is read from.
        scored: The hidden patches each loss term scores. (B, K)

    Returns:
        sums: Per term, the summed error. ()
    """
    with torch.autocast(cells.offset.device.type, dtype=torch.bfloat16):
        predictions = model(batch, visible, hidden, cells, kept)
    return summed_errors(predictions, batch, scored)


def step_terms(
    model: CrossSensorMAE,
    passes: list[tuple[dict[str, Tokens], Cells, list[str]]],
    mask_ratio: float,
    drop_ratio: float,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, Tensor]:
    """Return one step's loss terms, its gradients added up over every pass.

    Args:
        model: The model, on the device.
        passes: The batches the step is split into, each as the loader hands it.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        drop_ratio: The chance each instrument is left out of the grid.
        generator: What fixes the masks, on the device.
        device: Where the model runs.

    Returns:
        terms: Every loss term over the whole step, as if it were one batch.
    """
    # Every pass is drawn first, so each term is averaged over the whole step
    drawn, counts = [], defaultdict(float)
    for batch, cells, _ in passes:
        batch, _ = device_batch(batch, cells, device)
        visible, hidden, kept, scored = drawn_masks(
            batch, mask_ratio, drop_ratio, generator
        )
        drawn.append((visible, hidden, kept, scored))
        for term, patches in scored.items():
            counts[term] += float(patches.sum())
    sums = defaultdict(float)
    # One pass on the device at a time, its gradients added to the step's
    for (batch, cells, _), masks in zip(passes, drawn, strict=True):
        batch, cells = device_batch(batch, cells, device)
        held = batch_errors(model, batch, cells, *masks)
        loss_terms(held, counts)["loss"].backward()
        for term, value in held.items():
            sums[term] += value.detach()
    return loss_terms(sums, counts)
