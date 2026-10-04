"""What the model is trained to make small, under the names the paper gives it."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from architecture.grid import overlapping_boxes
from architecture.tokens import Tokens


def term_weights(
    batch: dict[str, Tokens],
    visible: dict[str, Tensor],
    hidden: dict[str, Tensor],
    kept: list[str],
) -> dict[tuple[str, str], Tensor]:
    """Return which patches each term reconstructs, before the model reads any.

    Args:
        batch: Each instrument's patches over the batch.
        visible: Which patches each encoder reads. (B, K)
        hidden: Which patches are predicted. (B, K)
        kept: Which instruments the grid is read from.

    Returns:
        weights: Per term and instrument asked, the hidden patches it counts: all of
            them under "umr" when the grid holds the instrument, under "cmr" only
            those a kept instrument reaches when it does not. (B, K)
    """
    weights = {}
    for asked, target in batch.items():
        if asked in kept:
            weights["umr", asked] = hidden[asked]
            continue
        readable = torch.zeros_like(hidden[asked])  # (B, K)
        for read in kept:
            source = batch[read]
            spans_delay = bool((target.position[..., 5] > 0).any()) or bool(
                (source.position[..., 5] > 0).any()
            )
            reaching = overlapping_boxes(
                target.position[:, :, None],
                source.position[:, None],
                3 if spans_delay else 2,
            )
            readable |= (reaching & visible[read][:, None]).any(dim=-1)  # (B, K)
        weights["cmr", asked] = hidden[asked] & readable  # (B, K)
    return weights


def reconstruction_sums(
    predictions: Mapping[str, Tensor],
    batch: dict[str, Tokens],
    weights: Mapping[tuple[str, str], Tensor],
) -> dict[tuple[str, str], Tensor]:
    """Return each term's squared error, summed over the patches it counts.

    Args:
        predictions: The predicted patches of each instrument.
        batch: Each instrument's patches over the batch.
        weights: The patches each term counts. (B, K)

    Returns:
        sums: Per term, every counted patch's mean squared error over its measured
            samples, added up. ()
    """
    sums = {}
    for (term, asked), weight in weights.items():
        tokens = batch[asked]
        counted = tokens.measured.to(tokens.values.dtype).expand_as(tokens.values)
        over = tuple(range(2, tokens.values.dim()))
        samples = counted.sum(dim=over).clamp(min=1)  # (B, K)
        error = (predictions[asked] - tokens.values) ** 2
        error = (error * counted).sum(dim=over) / samples  # (B, K)
        sums[term, asked] = (error * weight.to(error.dtype)).sum()  # ()
    return sums


def pooled_terms(
    sums: Mapping[tuple[str, str], Tensor], counts: Mapping[tuple[str, str], float]
) -> dict[str, Tensor]:
    """Return the UMR and CMR terms of the objective and their sum.

    Args:
        sums: Each term's squared error summed over the patches it counts. ()
        counts: How many patches each term counts, over every pass pooled.

    Returns:
        terms: "<term>/<sensor>" for every term drawn, and "loss", a term counting no
            patch nan and left out of the loss.
    """
    held = next(iter(sums.values()))
    terms = {}
    loss = held.new_zeros(())
    for (term, asked), summed in sums.items():
        value = torch.full((), torch.nan, device=held.device)
        if counts[term, asked]:
            value = summed / counts[term, asked]
            loss = loss + value
        terms[f"{term}/{asked}"] = value
    terms["loss"] = loss
    return terms
