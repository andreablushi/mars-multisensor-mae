"""What the model is trained to make small, under the names the paper gives it."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from architecture.tokens import Tokens


def cmr_patches(
    visible: dict[str, Tensor], hidden: dict[str, Tensor]
) -> dict[str, Tensor]:
    """Return the hidden patches CMR scores, those of a tile another instrument shows.

    Args:
        visible: Which patches each encoder reads. (B, K)
        hidden: Which patches are predicted. (B, K)

    Returns:
        scored: Each instrument's hidden patches of the tiles where another
            instrument has a visible patch. (B, K)
    """
    scored = {}
    for name in hidden:
        others = [visible[one].any(dim=1) for one in visible if one != name]
        read = torch.stack(others).any(dim=0)  # (B,)
        scored[name] = hidden[name] & read[:, None]
    return scored


def scored_counts(
    visible: dict[str, Tensor], hidden: dict[str, Tensor]
) -> dict[tuple[str, str], float]:
    """Return how many patches each term scores, keyed by term and instrument.

    Args:
        visible: Which patches each encoder reads. (B, K)
        hidden: Which patches are predicted. (B, K)

    Returns:
        counts: Under "umr" every hidden patch, under "cmr" those CMR scores.
    """
    return {
        (term, name): float(patches.sum())
        for term, scored in (("umr", hidden), ("cmr", cmr_patches(visible, hidden)))
        for name, patches in scored.items()
    }


def summed_error(prediction: Tensor, target: Tokens, scored: Tensor) -> Tensor:
    """Return the error of one instrument's prediction, summed over the patches scored.

    Args:
        prediction: The predicted patches. (B, K, *P)
        target: The instrument's patches over the batch.
        scored: Which patches are scored. (B, K)

    Returns:
        sum: Each scored patch's mean squared error over its measured samples,
            added up. ()
    """
    measured = target.measured.to(target.values.dtype).expand_as(target.values)
    samples = tuple(range(2, target.values.dim()))
    # Each patch's squared error, averaged over its measured samples
    error = ((prediction - target.values) ** 2 * measured).sum(dim=samples)
    error = error / measured.sum(dim=samples).clamp(min=1)  # (B, K)
    return (error * scored).sum()  # ()


def umr_errors(
    umr: Mapping[str, Tensor], batch: dict[str, Tokens], hidden: dict[str, Tensor]
) -> dict[str, Tensor]:
    """Return each instrument's UMR error, read from its own tokens.

    Args:
        umr: Each instrument's hidden patches read from its own tokens. (B, K, *P)
        batch: Each instrument's patches over the batch.
        hidden: Which patches are predicted. (B, K)

    Returns:
        sums: Each instrument's error, summed over its hidden patches. ()
    """
    return {name: summed_error(umr[name], batch[name], hidden[name]) for name in umr}


def cmr_errors(
    cmr: Mapping[str, Tensor],
    batch: dict[str, Tokens],
    visible: dict[str, Tensor],
    hidden: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Return each instrument's CMR error, read from every other instrument's tokens.

    Args:
        cmr: Each instrument's hidden patches read from the others' tokens. (B, K, *P)
        batch: Each instrument's patches over the batch.
        visible: Which patches each encoder reads. (B, K)
        hidden: Which patches are predicted. (B, K)

    Returns:
        sums: Each instrument's error, summed over the hidden patches CMR scores. ()
    """
    scored = cmr_patches(visible, hidden)
    return {name: summed_error(cmr[name], batch[name], scored[name]) for name in cmr}


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
