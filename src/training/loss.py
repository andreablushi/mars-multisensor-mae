"""What the model is trained to make small, under the names the paper gives it."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from architecture.grid import overlapping_boxes
from architecture.tokens import Tokens


def term_weights(
    batch: dict[str, Tokens], visible: dict[str, Tensor], hidden: dict[str, Tensor]
) -> dict[tuple[str, str], Tensor]:
    """Return which patches each term reconstructs, before the model reads any.

    Args:
        batch: Each instrument's patches over the batch.
        visible: Which patches each encoder reads. (B, K)
        hidden: Which patches are predicted. (B, K)

    Returns:
        weights: Per instrument asked and instrument read, the hidden patches it
            counts: all of them from itself, those another reaches from it. (B, K)
    """
    weights = {}
    for asked, target in batch.items():
        weights[asked, asked] = hidden[asked]
        for read, source in batch.items():
            if read == asked:
                continue
            spans_delay = bool((target.position[..., 5] > 0).any()) or bool(
                (source.position[..., 5] > 0).any()
            )
            reaching = overlapping_boxes(
                target.position[:, :, None],
                source.position[:, None],
                3 if spans_delay else 2,
            )
            readable = (reaching & visible[read][:, None]).any(dim=-1)  # (B, K)
            weights[asked, read] = hidden[asked] & readable  # (B, K)
    return weights


def reconstruction_sums(
    predictions: Mapping[tuple[str, str], Tensor],
    batch: dict[str, Tokens],
    weights: Mapping[tuple[str, str], Tensor],
) -> dict[tuple[str, str], Tensor]:
    """Return each term's squared error, summed over the patches it counts.

    Args:
        predictions: The predicted patches, by the instrument asked and the one read.
        batch: Each instrument's patches over the batch.
        weights: The patches each term counts. (B, K)

    Returns:
        sums: Per term, every counted patch's mean squared error over its measured
            samples, added up. ()
    """
    sums = {}
    for (asked, read), weight in weights.items():
        tokens = batch[asked]
        counted = tokens.measured.to(tokens.values.dtype).expand_as(tokens.values)
        over = tuple(range(2, tokens.values.dim()))
        samples = counted.sum(dim=over).clamp(min=1)  # (B, K)
        error = (predictions[asked, read] - tokens.values) ** 2
        error = (error * counted).sum(dim=over) / samples  # (B, K)
        sums[asked, read] = (error * weight.to(error.dtype)).sum()  # ()
    return sums


def pooled_terms(
    sums: Mapping[tuple[str, str], Tensor], counts: Mapping[tuple[str, str], float]
) -> dict[str, Tensor]:
    """Return the UMR and CMR terms of the objective and their sum.

    Args:
        sums: Each term's squared error summed over the patches it counts. ()
        counts: How many patches each term counts, over every pass pooled.

    Returns:
        terms: "umr/<sensor>", "cmr/<sensor>" and "loss", a term counting no patch
            nan and left out of the loss.
    """
    held = next(iter(sums.values()))
    nan = torch.full((), torch.nan, device=held.device)
    terms = {}
    loss = held.new_zeros(())
    for asked in dict.fromkeys(asked for asked, _ in sums):
        umr = nan
        if counts[asked, asked]:
            umr = sums[asked, asked] / counts[asked, asked]
            loss = loss + umr
        reads = [
            sums[asked, read] / counts[asked, read]
            for one, read in sums
            if one == asked and read != asked and counts[asked, read]
        ]
        cmr = nan
        if reads:
            cmr = torch.stack(reads).mean()
            loss = loss + cmr
        terms[f"umr/{asked}"], terms[f"cmr/{asked}"] = umr, cmr
    terms["loss"] = loss
    return terms
