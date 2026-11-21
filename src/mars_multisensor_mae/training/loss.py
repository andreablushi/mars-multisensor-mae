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


def patch_discrimination(
    prediction: Tensor, target: Tensor, scored: Tensor, temperature: float
) -> Tensor:
    """Return OlmoEarth's patch discrimination over the scored patches.

    Args:
        prediction: The predicted latents. (B, K, D)
        target: The patches' target latents. (B, K, D)
        scored: Which patches are scored. (B, K)
        temperature: What every cosine is divided by.

    Returns:
        error: The cross-entropy of each scored patch picking its own target among
            its tile's scored ones, averaged over the patches, times twice the
            temperature. ()
    """
    prediction = functional.normalize(prediction.float(), dim=-1)
    target = functional.normalize(target, dim=-1)
    logits = prediction @ target.mT / temperature  # (B, K, K)
    # A patch is told apart from its tile's other scored patches alone
    logits = logits.masked_fill(~scored.unsqueeze(1), float("-inf"))
    own = torch.arange(scored.shape[1], device=scored.device).expand_as(scored)
    return functional.cross_entropy(logits[scored], own[scored]) * 2 * temperature


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
    terms: Mapping[str, Tensor], mim_errors: Mapping[str, Tensor]
) -> dict[str, Tensor]:
    """Return every term and the loss, the latent and pixel sums plus the mean MIM.

    Args:
        terms: Each latent and pixel term, keyed "<term>/<instrument>", the pixel
            ones training their heads alone. ()
        mim_errors: Each aligned pair's MIM error, none when nothing is aligned. ()

    Returns:
        terms: The same terms, "mim/<pair>" for every MIM term, and "loss". ()
    """
    loss = sum(terms.values())
    if mim_errors:
        loss = loss + sum(mim_errors.values()) / len(mim_errors)
    return (
        dict(terms)
        | {f"mim/{pair}": error for pair, error in mim_errors.items()}
        | {"loss": loss}
    )
