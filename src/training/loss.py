"""What the model is trained to make small."""

from __future__ import annotations

import torch
from torch import Tensor

from architecture.mae import Reconstruction
from architecture.tokens import Tokens


def reconstruction_loss(
    prediction: Tensor, values: Tensor, valid: Tensor, weight: Tensor
) -> Tensor:
    """Return how far the predicted patches stand from the true ones.

    Args:
        prediction: The predicted patches. (B, K, *P)
        values: The true ones, as the model was handed them. (B, K, *P)
        valid: Whether each sample of a patch is a measurement, broadcastable
            to the values. (B, K, *P')
        weight: How much each patch counts. (B, K)

    Returns:
        loss: The mean squared error over the measured samples of the counted
            patches, each patch first centred and scaled by the mean and
            deviation of its own measured samples. Zero when none counts.
    """
    counted = valid.to(values.dtype).expand_as(values)  # (B, K, *P)
    over = tuple(range(2, values.dim()))
    spread = (*values.shape[:2], *([1] * len(over)))
    samples = counted.sum(dim=over).clamp(min=1)  # (B, K)
    mean = ((values * counted).sum(dim=over) / samples).reshape(spread)  # (B, K, 1...)
    variance = ((values - mean) ** 2 * counted).sum(dim=over) / samples  # (B, K)
    target = (values - mean) / (variance.reshape(spread) + 1e-6).sqrt()  # (B, K, *P)
    error = ((prediction - target) ** 2 * counted).sum(dim=over) / samples  # (B, K)
    weight = weight.to(values.dtype)  # (B, K)
    return (error * weight).sum() / weight.sum().clamp(min=1)  # ()


def uniformity_loss(directions: Tensor, present: Tensor) -> Tensor:
    """Return how far the tokens of a batch are from spreading over the sphere.

    Args:
        directions: Unit vectors, every instrument's tokens side by side.
            (B, K, L)
        present: Which of them count. (B, K)

    Returns:
        loss: The mean absolute dot product between each token and the one in
            the same slot of the next feature of the batch, which two
            independent draws from a uniform sphere would make zero.
    """
    rolled = directions.roll(1, dims=0)  # (B, K, L)
    weight = (present & present.roll(1, dims=0)).to(directions.dtype)  # (B, K)
    agreement = (directions * rolled).sum(dim=-1).abs()  # (B, K)
    return (agreement * weight).sum() / weight.sum().clamp(min=1)  # ()


def total_loss(
    reconstruction: Reconstruction, batch: dict[str, Tokens], uniformity_weight: float
) -> dict[str, Tensor]:
    """Return every term of the loss, and their weighted sum under "loss".

    Args:
        reconstruction: What the masked pass predicted, and from what.
        batch: What it was handed.
        uniformity_weight: How much the uniformity term weighs against the
            reconstruction.

    Returns:
        terms: Under "reconstruction/<instrument>" the error of its hidden
            patches read from its own tokens, under "cross/<instrument>" that
            error read from each other instrument's tokens, averaged, under
            "uniformity" the spread of every visible token, and under "loss"
            the reconstruction terms summed plus the weighted uniformity.
    """
    terms = {}
    total = torch.zeros((), device=next(iter(batch.values())).values.device)  # ()
    for asked, tokens in batch.items():
        hidden = tokens.present & ~reconstruction.visible[asked]  # (B, K)
        own = reconstruction_loss(
            reconstruction.predictions[asked, asked],
            tokens.values,
            tokens.valid,
            hidden,
        )  # ()
        others = []
        for read in batch:
            if read == asked:
                continue
            readable = reconstruction.visible[read].any(dim=1, keepdim=True)  # (B, 1)
            others.append(
                reconstruction_loss(
                    reconstruction.predictions[asked, read],
                    tokens.values,
                    tokens.valid,
                    hidden & readable,
                )
            )
        cross = torch.stack(others).mean() if others else total  # ()
        terms[f"reconstruction/{asked}"] = own
        terms[f"cross/{asked}"] = cross
        total = total + own + cross
    directions = torch.cat([reconstruction.tokens[name] for name in batch], dim=1)
    visible = torch.cat([reconstruction.visible[name] for name in batch], dim=1)
    terms["uniformity"] = uniformity_loss(directions, visible)  # ()
    terms["loss"] = total + uniformity_weight * terms["uniformity"]  # ()
    return terms
