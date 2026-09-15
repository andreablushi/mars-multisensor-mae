"""What the model is trained to make small, under the names the paper gives it."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional

from architecture.components.crossattention_fusion import instrument_vector
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


def mutual_information(
    tokens: dict[str, Tensor], present: dict[str, Tensor], temperature: float
) -> Tensor:
    """Return the contrastive term holding a feature's instruments to each other.

    Args:
        tokens: Each sensor's tokens, none hidden. (B, K, D)
        present: Which of each instrument's slots hold a patch. (B, K)
        temperature: What the similarities are divided by.

    Returns:
        loss: The paper's L_MIM, over every ordered pair of sensors, its positive kept.
    """
    vectors = {
        name: functional.normalize(instrument_vector(held, present[name]), dim=-1)
        for name, held in tokens.items()
    }  # (B, D)
    read = {name: held.any(dim=1) for name, held in present.items()}  # (B,)
    terms = []
    for asked, query in vectors.items():
        for against, key in vectors.items():
            if against == asked:
                continue
            similarity = query @ key.T / temperature  # (B, B)
            counted = read[asked] & read[against]  # (B,)
            floor = torch.finfo(similarity.dtype).min
            against_counted = similarity.masked_fill(
                ~counted.unsqueeze(0), floor
            )  # (B, B)
            denominator = against_counted.logsumexp(dim=1)  # (B,)
            error = denominator - similarity.diagonal()  # (B,)
            terms.append(
                torch.where(counted, error, 0.0).sum() / counted.sum().clamp(min=1)
            )
    if not terms:
        return torch.zeros((), device=next(iter(tokens.values())).device)  # ()
    return torch.stack(terms).mean()  # ()


def csmae_loss(
    reconstruction: Reconstruction, batch: dict[str, Tokens], temperature: float
) -> dict[str, Tensor]:
    """Return every term of the objective, and their sum under "loss".

    Args:
        reconstruction: What the masked pass predicted, and from what.
        batch: What it was handed.
        temperature: What the contrastive term divides its similarities by.

    Returns:
        terms: "umr/<sensor>", "cmr/<sensor>", "mim", and "loss" summing them all.
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
            readable = batch[read].visible.any(dim=1, keepdim=True)  # (B, 1)
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
    terms["mim"] = mutual_information(
        reconstruction.tokens,
        {name: tokens.present for name, tokens in batch.items()},
        temperature,
    )  # ()
    terms["loss"] = total + terms["mim"]  # ()
    return terms
