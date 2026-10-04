"""What the model is trained to make small, under the names the paper gives it."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from architecture.grid import overlapping_boxes
from architecture.tokens import Tokens


def scored_patches(
    batch: dict[str, Tokens],
    visible: dict[str, Tensor],
    hidden: dict[str, Tensor],
    kept: list[str],
) -> dict[tuple[str, str], Tensor]:
    """Return the hidden patches each loss term scores, keyed by term and instrument.

    Args:
        batch: Each instrument's patches over the batch.
        visible: Which patches each encoder reads. (B, K)
        hidden: Which patches are predicted. (B, K)
        kept: Which instruments the grid is read from.

    Returns:
        scored: Under "umr" every hidden patch of an instrument the grid holds,
            under "cmr" those of one it leaves out that a kept one reaches. (B, K)
    """
    scored = {}
    for name, target in batch.items():
        if name in kept:
            scored["umr", name] = hidden[name]
            continue
        reached = torch.zeros_like(hidden[name])  # (B, K)
        for source_name in kept:
            source = batch[source_name]
            # Surface patches are matched on the ground alone, a sounder in delay too
            sounder = (target.position[..., 5] > 0).any() or (
                source.position[..., 5] > 0
            ).any()
            overlapping = overlapping_boxes(
                target.position[:, :, None],
                source.position[:, None],
                3 if sounder else 2,
            )  # (B, K, K')
            reached |= (overlapping & visible[source_name][:, None]).any(dim=-1)
        scored["cmr", name] = hidden[name] & reached  # (B, K)
    return scored


def summed_errors(
    predictions: Mapping[str, Tensor],
    batch: dict[str, Tokens],
    scored: Mapping[tuple[str, str], Tensor],
) -> dict[tuple[str, str], Tensor]:
    """Return each term's error, summed over the patches it scores.

    Args:
        predictions: The predicted patches of each instrument. (B, K, *P)
        batch: Each instrument's patches over the batch.
        scored: The patches each term scores. (B, K)

    Returns:
        sums: Per term, each scored patch's mean squared error over its measured
            samples, added up. ()
    """
    sums = {}
    for (term, name), patches in scored.items():
        target = batch[name]
        measured = target.measured.to(target.values.dtype).expand_as(target.values)
        samples = tuple(range(2, target.values.dim()))
        # Each patch's squared error, averaged over its measured samples
        error = ((predictions[name] - target.values) ** 2 * measured).sum(dim=samples)
        error = error / measured.sum(dim=samples).clamp(min=1)  # (B, K)
        sums[term, name] = (error * patches).sum()  # ()
    return sums


def loss_terms(
    sums: Mapping[tuple[str, str], Tensor], counts: Mapping[tuple[str, str], float]
) -> dict[str, Tensor]:
    """Return each term's mean error over the patches it scored, and the loss.

    Args:
        sums: Each term's error, summed over the patches it scored. ()
        counts: How many patches each term scored, over every pass pooled.

    Returns:
        terms: "<term>/<instrument>" for every term, nan where it scored no patch,
            and "loss", the sum of every term that scored one.
    """
    held = next(iter(sums.values()))
    terms = {}
    loss = held.new_zeros(())
    for (term, name), summed in sums.items():
        mean = torch.full((), torch.nan, device=held.device)
        if counts[term, name]:
            mean = summed / counts[term, name]
            loss = loss + mean
        terms[f"{term}/{name}"] = mean
    terms["loss"] = loss
    return terms
