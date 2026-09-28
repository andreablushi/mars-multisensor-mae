"""What the model is trained to make small, under the names the paper gives it."""

from __future__ import annotations

from collections.abc import Collection

import torch
from torch import Tensor

from architecture.grid import overlapping_boxes
from architecture.mae import Reconstruction
from architecture.tokens import Tokens


def patch_targets(
    values: Tensor, valid: Tensor, normalised: bool
) -> tuple[Tensor, Tensor]:
    """Return the patches the predictions are measured against.

    Args:
        values: The patches, as the model was handed them. (B, K, *P)
        valid: Whether each sample is a measurement, broadcastable to them. (B, K, *P')
        normalised: Whether each patch is centred and scaled by its own samples.

    Returns:
        target: The patches, each of zero mean and unit deviation if normalised,
            else on the instrument's own scale. (B, K, *P)
        counted: Whether each sample is a measurement, spread over them. (B, K, *P)
    """
    counted = valid.to(values.dtype).expand_as(values)  # (B, K, *P)
    if not normalised:
        return values, counted
    over = tuple(range(2, values.dim()))
    spread = (*values.shape[:2], *([1] * len(over)))
    samples = counted.sum(dim=over).clamp(min=1)  # (B, K)
    mean = ((values * counted).sum(dim=over) / samples).reshape(spread)  # (B, K, 1...)
    variance = ((values - mean) ** 2 * counted).sum(dim=over) / samples  # (B, K)
    deviation = (variance.reshape(spread) + 1e-6).sqrt()  # (B, K, 1...)
    return (values - mean) / deviation, counted


def reconstruction_error(
    prediction: Tensor, values: Tensor, valid: Tensor, weight: Tensor, normalised: bool
) -> Tensor:
    """Return how far the predicted patches stand from the true ones.

    Args:
        prediction: The predicted patches. (B, K, *P)
        values: The true ones, as the model was handed them. (B, K, *P)
        valid: Whether each sample is a measurement, broadcastable to them. (B, K, *P')
        weight: How much each patch counts. (B, K)
        normalised: Whether each true patch is normalised by its own samples.

    Returns:
        error: The mean squared error over the counted patches, zero where none is.
    """
    target, counted = patch_targets(values, valid, normalised)  # (B, K, *P)
    over = tuple(range(2, values.dim()))
    samples = counted.sum(dim=over).clamp(min=1)  # (B, K)
    error = ((prediction - target) ** 2 * counted).sum(dim=over) / samples  # (B, K)
    weight = weight.to(values.dtype)  # (B, K)
    return (error * weight).sum() / weight.sum().clamp(min=1)  # ()


def umr_loss(prediction: Tensor, tokens: Tokens, normalised: bool) -> Tensor:
    """Return masked reconstruction error from the same instrument.

    Args:
        prediction: The predicted patches. (B, K, *P)
        tokens: The instrument's patches and visibility mask.
        normalised: Whether each true patch is normalised by its own samples.

    Returns:
        error: Mean squared error over hidden measured patches.
    """
    hidden = tokens.present & ~tokens.visible
    return reconstruction_error(
        prediction, tokens.values, tokens.valid, hidden, normalised
    )


def cmr_loss(
    reconstruction: Reconstruction,
    batch: dict[str, Tokens],
    asked: str,
    normalised: bool,
) -> Tensor:
    """Return masked reconstruction error from the other instruments.

    Args:
        reconstruction: The predictions made from each source instrument.
        batch: The patches and masks of every instrument.
        asked: The instrument whose patches are reconstructed.
        normalised: Whether each true patch is normalised by its own samples.

    Returns:
        error: Mean error across the other instruments, including empty readers.
    """
    target = batch[asked]
    hidden = target.present & ~target.visible
    errors = []
    for read, source in batch.items():
        if read == asked:
            continue
        reaching = overlapping_boxes(
            target.position[:, :, None], source.position[:, None], 2
        )
        readable = (reaching & source.visible[:, None]).any(dim=-1)
        errors.append(
            reconstruction_error(
                reconstruction.predictions[asked, read],
                target.values,
                target.valid,
                hidden & readable,
                normalised,
            )
        )
    return torch.stack(errors).mean() if errors else target.values.new_zeros(())


def csmae_loss(
    reconstruction: Reconstruction,
    batch: dict[str, Tokens],
    unnormalised_patches: Collection[str],
) -> dict[str, Tensor]:
    """Return the UMR and CMR terms of the objective and their sum.

    Args:
        reconstruction: What the masked pass predicted.
        batch: What it was handed.
        unnormalised_patches: The instruments whose targets keep their own scale.

    Returns:
        terms: "umr/<sensor>", "cmr/<sensor>", and "loss".
    """
    terms = {}
    total = next(iter(batch.values())).values.new_zeros(())
    for asked, tokens in batch.items():
        normalised = asked not in unnormalised_patches
        umr = umr_loss(reconstruction.predictions[asked, asked], tokens, normalised)
        cmr = cmr_loss(reconstruction, batch, asked, normalised)
        terms[f"umr/{asked}"] = umr
        terms[f"cmr/{asked}"] = cmr
        total = total + umr + cmr
    terms["loss"] = total  # ()
    return terms
