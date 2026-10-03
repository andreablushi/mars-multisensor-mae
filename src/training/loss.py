"""What the model is trained to make small, under the names the paper gives it."""

from __future__ import annotations

import torch
from torch import Tensor

from architecture.grid import overlapping_boxes
from architecture.mae import Reconstruction
from architecture.tokens import Tokens


def reconstruction_error(
    prediction: Tensor, values: Tensor, measured: Tensor, weight: Tensor
) -> Tensor:
    """Return the mean squared error over the counted patches.

    Args:
        prediction: The predicted patches, on the dataset's scale. (B, K, *P)
        values: The true ones, as the model was handed them. (B, K, *P)
        measured: Whether each sample is a measurement, broadcastable. (B, K, *P')
        weight: How much each patch counts. (B, K)

    Returns:
        error: Each counted patch's mean squared error over its measured samples,
            averaged over the patches by weight. Nan where none is counted.
    """
    counted = measured.to(values.dtype).expand_as(values)  # (B, K, *P)
    over = tuple(range(2, values.dim()))
    samples = counted.sum(dim=over).clamp(min=1)  # (B, K)
    error = ((prediction - values) ** 2 * counted).sum(dim=over) / samples  # (B, K)
    weight = weight.to(values.dtype)  # (B, K)
    if not weight.any():
        return error.new_tensor(torch.nan)  # ()
    return (error * weight).sum() / weight.sum()  # ()


def umr_loss(prediction: Tensor, tokens: Tokens, visible: Tensor) -> Tensor:
    """Return masked reconstruction error from the same instrument.

    Args:
        prediction: The predicted patches. (B, K, *P)
        tokens: The instrument's patches.
        visible: Which of them its encoder read. (B, K)

    Returns:
        error: Mean squared error over hidden measured patches.
    """
    hidden = tokens.measured_slots & ~visible
    return reconstruction_error(prediction, tokens.values, tokens.measured, hidden)


def cmr_loss(
    reconstruction: Reconstruction,
    batch: dict[str, Tokens],
    visible: dict[str, Tensor],
    asked: str,
) -> Tensor:
    """Return masked reconstruction error from the other instruments.

    Args:
        reconstruction: The predictions made from each source instrument.
        batch: The patches and masks of every instrument.
        visible: Which patches each encoder read. (B, K)
        asked: The instrument whose patches are reconstructed.

    Returns:
        error: Mean error across the other instruments that read any, else nan.
    """
    target = batch[asked]
    hidden = target.measured_slots & ~visible[asked]
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
        readable = (reaching & visible[read][:, None]).any(dim=-1)
        errors.append(
            reconstruction_error(
                reconstruction.predictions[asked, read],
                target.values,
                target.measured,
                hidden & readable,
            )
        )
    return (
        torch.stack(errors).nanmean() if errors else target.values.new_tensor(torch.nan)
    )


def csmae_loss(
    reconstruction: Reconstruction,
    batch: dict[str, Tokens],
    visible: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Return the UMR and CMR terms of the objective and their sum.

    Args:
        reconstruction: What the masked pass predicted.
        batch: What it was handed.
        visible: Which patches each encoder read. (B, K)

    Returns:
        terms: "umr/<sensor>", "cmr/<sensor>", and "loss".
    """
    terms = {}
    total = next(iter(batch.values())).values.new_zeros(())
    for asked, tokens in batch.items():
        umr = umr_loss(reconstruction.predictions[asked, asked], tokens, visible[asked])
        cmr = cmr_loss(reconstruction, batch, visible, asked)
        terms[f"umr/{asked}"] = umr
        terms[f"cmr/{asked}"] = cmr
        total = total + umr.nan_to_num() + cmr.nan_to_num()
    terms["loss"] = total  # ()
    return terms
