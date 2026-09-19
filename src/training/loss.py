"""What the model is trained to make small, under the names the paper gives it."""

from __future__ import annotations

import torch
from torch import Tensor

from architecture.mae import Reconstruction
from architecture.models import Tokens
from dataset.patches import normalize_patches


def reconstruction_error(
    prediction: Tensor, values: Tensor, valid: Tensor, weight: Tensor
) -> Tensor:
    """Return how far the predicted patches stand from the true ones.

    Args:
        prediction: The predicted patches. (B, K, *P)
        values: The true ones, as the model was handed them. (B, K, *P)
        valid: Whether each sample is a measurement, broadcastable to them. (B, K, *P')
        weight: How much each patch counts. (B, K)

    Returns:
        error: The mean squared error over the counted patches, zero where none is.
    """
    target, counted, *_ = normalize_patches(values, valid)  # (B, K, *P)
    over = tuple(range(2, values.dim()))
    samples = counted.sum(dim=over).clamp(min=1)  # (B, K)
    error = ((prediction - target) ** 2 * counted).sum(dim=over) / samples  # (B, K)
    weight = weight.to(values.dtype)  # (B, K)
    return (error * weight).sum() / weight.sum().clamp(min=1)  # ()


def csmae_loss(
    reconstruction: Reconstruction, batch: dict[str, Tokens]
) -> dict[str, Tensor]:
    """Return every term of the objective, and their sum under "loss".

    Args:
        reconstruction: What the masked pass predicted, and the grids it read into.
        batch: What it was handed.

    Returns:
        terms: "umr/<sensor>", "cmr/<sensor>", and "loss".
    """
    terms = {}
    total = torch.zeros((), device=next(iter(batch.values())).values.device)  # ()
    for asked, tokens in batch.items():
        hidden = tokens.present & ~tokens.visible  # (B, K)
        umr = reconstruction_error(
            reconstruction.predictions[asked, asked],
            tokens.values,
            tokens.valid,
            hidden,
        )  # ()
        others = []
        for read in batch:
            if read == asked:
                continue
            readable = reconstruction.grids[read].occupied.any(
                dim=1, keepdim=True
            )  # (B, 1)
            others.append(
                reconstruction_error(
                    reconstruction.predictions[asked, read],
                    tokens.values,
                    tokens.valid,
                    hidden & readable,
                )
            )
        cmr = torch.stack(others).mean() if others else total  # ()
        terms[f"umr/{asked}"] = umr
        terms[f"cmr/{asked}"] = cmr
        total = total + umr + cmr
    terms["loss"] = total  # ()
    return terms
