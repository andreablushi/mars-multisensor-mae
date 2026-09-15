"""What the model is trained to make small, under the names the paper gives it."""

from __future__ import annotations

import torch
from torch import Tensor

from architecture.mae import Reconstruction
from architecture.models import FeatureGrid, Tokens
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


def consistency(whole: FeatureGrid, part: FeatureGrid) -> Tensor:
    """Return how far a grid read from some instruments stands from the grid of all.

    Args:
        whole: The grid every instrument the feature holds was read into.
        part: The grid one of them alone was read into.

    Returns:
        loss: Half one minus their agreement, over the cells both of them reach.
    """
    counted = (whole.occupied & part.occupied).to(whole.values.dtype)  # (B, Q)
    agreement = (whole.values * part.values).sum(dim=-1)  # (B, Q)
    error = (1 - agreement) / 2  # (B, Q)
    return (error * counted).sum() / counted.sum().clamp(min=1)  # ()


def uniformity(grid: FeatureGrid) -> Tensor:
    """Return how far the cells stand from spread evenly over the space they live in.

    Two cells drawn from the training set at random stand orthogonal where the
    cells are spread, so the batch is rolled to pair each cell with one of
    another feature and their agreement is what is made small.

    Args:
        grid: The grid every instrument the feature holds was read into.

    Returns:
        loss: The mean agreement of those pairs, without regard to its sign.
    """
    against = grid.values.roll(1, dims=0)  # (B, Q, D)
    paired = (grid.occupied & grid.occupied.roll(1, dims=0)).to(grid.values.dtype)
    agreement = (grid.values * against).sum(dim=-1).abs()  # (B, Q)
    return (agreement * paired).sum() / paired.sum().clamp(min=1)  # ()


def csmae_loss(
    reconstruction: Reconstruction,
    batch: dict[str, Tokens],
    consistency_weight: float,
    uniformity_weight: float,
) -> dict[str, Tensor]:
    """Return every term of the objective, and their sum under "loss".

    Args:
        reconstruction: What the masked pass predicted, and the grids it read into.
        batch: What it was handed.
        consistency_weight: What one instrument's grid agreeing with the whole counts.
        uniformity_weight: What the cells standing apart from each other counts.

    Returns:
        terms: "umr/<sensor>", "cmr/<sensor>", "consistency", "uniformity", and "loss".
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
    held = [
        consistency(reconstruction.grid, one) for one in reconstruction.grids.values()
    ]
    terms["consistency"] = torch.stack(held).mean() if held else total  # ()
    terms["uniformity"] = uniformity(reconstruction.grid)  # ()
    terms["loss"] = (
        total
        + consistency_weight * terms["consistency"]
        + uniformity_weight * terms["uniformity"]
    )  # ()
    return terms
