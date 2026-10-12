"""What the model is trained to make small, under the names the paper gives it."""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor

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
    """Return each instrument's CMR error, read from its sources' tokens.

    Args:
        predictions: Each instrument's patches read from its sources' tokens.
            (B, K, *P)
        batch: Each instrument's patches over the batch.
        hidden: Which patches are predicted. (B, K)

    Returns:
        errors: Each instrument's error over its hidden patches. ()
    """
    return {name: mse(predictions[name], batch[name], hidden[name]) for name in batch}


def total(
    umr_errors: Mapping[str, Tensor], cmr_errors: Mapping[str, Tensor]
) -> dict[str, Tensor]:
    """Return every term and the loss, their sum.

    Args:
        umr_errors: Each instrument's UMR error. ()
        cmr_errors: Each instrument's CMR error. ()

    Returns:
        terms: "<term>/<instrument>" for every term, and "loss". ()
    """
    terms = {
        f"{term}/{name}": error
        for term, errors in (("umr", umr_errors), ("cmr", cmr_errors))
        for name, error in errors.items()
    }
    return terms | {"loss": sum(terms.values())}
