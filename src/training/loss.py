"""What the model is trained to make small, under the names the paper gives it."""

from __future__ import annotations

import torch
from torch import Tensor

from architecture.grid import overlapping_boxes
from architecture.mae import Reconstruction
from architecture.tokens import Tokens


def reconstruction_error(
    prediction: Tensor, values: Tensor, valid: Tensor, weight: Tensor
) -> Tensor:
    """Return the share of the true patches' energy the prediction misses.

    Args:
        prediction: The predicted patches, on the instrument's scale. (B, K, *P)
        values: The true ones, as the model was handed them. (B, K, *P)
        valid: Whether each sample is a measurement, broadcastable to them. (B, K, *P')
        weight: How much each patch counts. (B, K)

    Returns:
        error: The squared error over the counted patches, against the squared
            values of those same patches, so predicting zero scores 1. Zero where
            none is counted.
    """
    counted = valid.to(values.dtype).expand_as(values)  # (B, K, *P)
    over = tuple(range(2, values.dim()))
    samples = counted.sum(dim=over).clamp(min=1)  # (B, K)
    error = ((prediction - values) ** 2 * counted).sum(dim=over) / samples  # (B, K)
    energy = (values**2 * counted).sum(dim=over) / samples  # (B, K)
    weight = weight.to(values.dtype)  # (B, K)
    return (error * weight).sum() / (energy * weight).sum().clamp(min=1e-6)  # ()


def umr_loss(prediction: Tensor, tokens: Tokens) -> Tensor:
    """Return masked reconstruction error from the same instrument.

    Args:
        prediction: The predicted patches. (B, K, *P)
        tokens: The instrument's patches and visibility mask.

    Returns:
        error: Mean squared error over hidden measured patches.
    """
    hidden = tokens.present & ~tokens.visible
    return reconstruction_error(prediction, tokens.values, tokens.valid, hidden)


def cmr_loss(
    reconstruction: Reconstruction, batch: dict[str, Tokens], asked: str
) -> Tensor:
    """Return masked reconstruction error from the other instruments.

    Args:
        reconstruction: The predictions made from each source instrument.
        batch: The patches and masks of every instrument.
        asked: The instrument whose patches are reconstructed.

    Returns:
        error: Mean error across the other instruments, including empty readers.
    """
    target = batch[asked]
    hidden = target.present & ~target.visible
    errors = []
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
        readable = (reaching & source.visible[:, None]).any(dim=-1)
        errors.append(
            reconstruction_error(
                reconstruction.predictions[asked, read],
                target.values,
                target.valid,
                hidden & readable,
            )
        )
    return torch.stack(errors).mean() if errors else target.values.new_zeros(())


def csmae_loss(
    reconstruction: Reconstruction, batch: dict[str, Tokens]
) -> dict[str, Tensor]:
    """Return the UMR and CMR terms of the objective and their mean.

    Args:
        reconstruction: What the masked pass predicted.
        batch: What it was handed.

    Returns:
        terms: "umr/<sensor>", "cmr/<sensor>", and "loss".
    """
    terms = {}
    total = next(iter(batch.values())).values.new_zeros(())
    for asked, tokens in batch.items():
        umr = umr_loss(reconstruction.predictions[asked, asked], tokens)
        cmr = cmr_loss(reconstruction, batch, asked)
        terms[f"umr/{asked}"] = umr
        terms[f"cmr/{asked}"] = cmr
        total = total + umr + cmr
    terms["loss"] = total / (2 * len(batch))  # ()
    return terms
