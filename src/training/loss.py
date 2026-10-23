"""What the model is trained to make small, under the names the paper gives it."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations

import torch
from torch import Tensor
from torch.nn import functional

from architecture.tokens import Tokens


def mse(prediction: Tensor, target: Tokens, scored: Tensor) -> Tensor:
    """Return the mean squared error over the scored patches.

    Args:
        prediction: The predicted patches. (B, K, *P)
        target: The instrument's patches over the batch.
        scored: Which patches are scored. (B, K)

    Returns:
        error: Each scored patch's squared error over its measured samples,
            averaged over the patches. ()
    """
    measured = target.measured.to(target.values.dtype).expand_as(target.values)
    samples = tuple(range(2, target.values.dim()))
    # Each patch's squared error, averaged over its measured samples
    error = ((prediction - target.values) ** 2 * measured).sum(dim=samples)
    error = error / measured.sum(dim=samples).clamp(min=1)  # (B, K)
    return error[scored].mean()  # ()


def umr(
    predictions: Mapping[str, Tensor],
    batch: Mapping[str, Tokens],
    hidden: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Return each instrument's UMR error, read from its own tokens.

    Args:
        predictions: Each instrument's patches read from its own tokens. (B, K, *P)
        batch: Each instrument's patches over the batch.
        hidden: Which patches are predicted. (B, K)

    Returns:
        errors: Each instrument's error over its hidden patches. ()
    """
    return {name: mse(predictions[name], batch[name], hidden[name]) for name in batch}


def cmr(
    predictions: Mapping[str, Tensor],
    batch: Mapping[str, Tokens],
    hidden: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Return each instrument's CMR error, read from every other instrument's tokens.

    Args:
        predictions: Each instrument's patches read from the others' tokens.
            (B, K, *P)
        batch: Each instrument's patches over the batch.
        hidden: Which patches are predicted. (B, K)

    Returns:
        errors: Each instrument's error over its hidden patches. ()
    """
    return {name: mse(predictions[name], batch[name], hidden[name]) for name in batch}


def mim(
    pooled: Mapping[str, Tensor], aligned: Sequence[str], temperature: float
) -> dict[str, Tensor]:
    """Return CSMAE's NT-Xent of every pair of aligned instruments.

    Args:
        pooled: Each instrument's tile vector as one unit vector. (B, D)
        aligned: The instruments aligned, each pair its own term.
        temperature: What every cosine is divided by.

    Returns:
        errors: Each pair's error, keyed "<first>-<second>". ()
    """
    errors = {}
    for first, second in combinations(aligned, 2):
        vectors = torch.cat([pooled[first], pooled[second]])  # (2B, D)
        logits = vectors @ vectors.T / temperature  # (2B, 2B)
        # A vector is never its own negative, and its tile's other one is the positive
        logits.fill_diagonal_(float("-inf"))
        partner = torch.arange(len(vectors), device=vectors.device).roll(
            len(vectors) // 2
        )
        errors[f"{first}-{second}"] = functional.cross_entropy(logits, partner)
    return errors


def total(
    umr_errors: Mapping[str, Tensor],
    cmr_errors: Mapping[str, Tensor],
    mim_errors: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Return every term and the loss, the reconstruction sum plus the mean MIM.

    Args:
        umr_errors: Each instrument's UMR error. ()
        cmr_errors: Each instrument's CMR error. ()
        mim_errors: Each aligned pair's MIM error, none when nothing is aligned. ()

    Returns:
        terms: "<term>/<instrument>" or "mim/<pair>" for every term, and "loss". ()
    """
    terms = {
        f"{term}/{name}": error
        for term, errors in (("umr", umr_errors), ("cmr", cmr_errors))
        for name, error in errors.items()
    }
    loss = sum(terms.values())
    if mim_errors:
        loss = loss + sum(mim_errors.values()) / len(mim_errors)
    return (
        terms
        | {f"mim/{pair}": error for pair, error in mim_errors.items()}
        | {"loss": loss}
    )
