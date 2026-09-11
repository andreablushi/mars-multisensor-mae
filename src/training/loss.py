"""What the model is trained to make small, under the names the paper gives it."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional

from architecture.fusion import instrument_vector
from architecture.mae import Reconstruction
from architecture.tokens import Tokens


def reconstruction_error(
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
        error: The mean squared error over the measured samples of the counted
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


def mutual_information(
    tokens: dict[str, Tensor], visible: dict[str, Tensor], temperature: float
) -> Tensor:
    """Return the contrastive term holding a feature's instruments to each other.

    Args:
        tokens: Each instrument's tokens as the cross-sensor encoder hands
            them, keyed as ODE names it. (B, K, D)
        visible: Which of each instrument's tokens its encoder read. (B, K)
        temperature: What the similarities are divided by.

    Returns:
        loss: The paper's L_MIM, averaged over every ordered pair of
            instruments: for each feature, minus the log of its own pair of
            vectors' similarity over that of its vector against every other
            feature of the batch. One vector stands for a feature under an
            instrument, the average of the tokens that instrument read of it.
            A feature holding no visible token of either instrument of a pair
            is neither a query nor a negative of it. Zero where the model reads
            one instrument alone.
    """
    vectors = {
        name: functional.normalize(instrument_vector(held, visible[name]), dim=-1)
        for name, held in tokens.items()
    }  # (B, D)
    read = {name: held.any(dim=1) for name, held in visible.items()}  # (B,)
    terms = []
    for asked, query in vectors.items():
        for against, key in vectors.items():
            if against == asked:
                continue
            similarity = query @ key.T / temperature  # (B, B)
            paired = read[asked] & read[against]  # (B,)
            others = paired.unsqueeze(0) & ~torch.eye(
                paired.shape[0], dtype=torch.bool, device=paired.device
            )  # (B, B)
            floor = torch.finfo(similarity.dtype).min
            against_others = similarity.masked_fill(~others, floor)  # (B, B)
            denominator = against_others.logsumexp(dim=1)  # (B,)
            counted = paired & others.any(dim=1)  # (B,)
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
        terms: Under "umr/<instrument>" the uni-modal reconstruction of its
            hidden patches, read from its own visible tokens, under
            "cmr/<instrument>" the cross-modal reconstruction of those same
            patches read from each other instrument's, averaged, under "mim"
            the contrastive term, and under "loss" every reconstruction term
            summed plus it.
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
        {name: tokens.visible for name, tokens in batch.items()},
        temperature,
    )  # ()
    terms["loss"] = total + terms["mim"]  # ()
    return terms
