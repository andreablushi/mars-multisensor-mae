"""What the model is trained to make small: the reconstruction error and MIM."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations

import torch
from torch import Tensor
from torch.nn import functional

from mars_multisensor_mae.architecture.models.tokens import Tokens


def reconstruction_error(prediction: Tensor, target: Tokens, scored: Tensor) -> Tensor:
    """Return the root mean squared error over the scored patches.

    Args:
        prediction: The predicted patches. (B, K, *P)
        target: The instrument's patches over the batch.
        scored: Which patches are scored. (B, K)

    Returns:
        error: Each scored patch's squared error over its measured samples,
            averaged over the patches, then rooted. ()
    """
    measured = target.measured.to(target.values.dtype).expand_as(target.values)
    samples = tuple(range(2, target.values.dim()))
    # Each patch's squared error, averaged over its measured samples
    error = ((prediction - target.values) ** 2 * measured).sum(dim=samples)
    error = error / measured.sum(dim=samples).clamp(min=1)  # (B, K)
    error = error[scored].mean()  # ()
    return error.sqrt()


def mutual_information_maximization_losses(
    pooled: Mapping[str, Tensor], aligned: Sequence[str], temperature: float
) -> dict[str, Tensor]:
    """Return CSMAE's mutual information maximization loss, an NT-Xent, of every pair.

    Args:
        pooled: Each instrument's projected tile vector as one unit vector. (B, P)
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


def terms_with_total_loss(
    terms: Mapping[str, Tensor], mutual_information: Mapping[str, Tensor]
) -> dict[str, Tensor]:
    """Return every term and the loss, the reconstruction sum plus the mean MIM.

    Args:
        terms: Each reconstruction term, keyed "<term>/<instrument>". ()
        mutual_information: Each aligned pair's MIM loss, none when nothing is
            aligned. ()

    Returns:
        terms: The same terms, "mim/<pair>" for every MIM term, and "loss". ()
    """
    loss = sum(terms.values())
    if mutual_information:
        loss = loss + sum(mutual_information.values()) / len(mutual_information)
    return (
        dict(terms)
        | {f"mim/{pair}": error for pair, error in mutual_information.items()}
        | {"loss": loss}
    )
