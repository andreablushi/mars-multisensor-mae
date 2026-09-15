"""Hiding patches from their encoder, as the cross-sensor correspondences do."""

from __future__ import annotations

from dataclasses import replace

import torch

from architecture.models import Tokens


def random_correspondence(
    batch: dict[str, Tokens], ratio: float, generator: torch.Generator
) -> dict[str, Tokens]:
    """Return the batch with each instrument's patches hidden on a draw of its own.

    Args:
        batch: Each instrument's patches over the batch, keyed as ODE names it.
        ratio: The share of a sensor's present patches to hide, rounded down.
        generator: What fixes the draw, on the batch's own device.

    Returns:
        masked: The same patches, `visible` now what each encoder may read.
    """
    masked = {}
    for name, tokens in batch.items():
        present = tokens.present  # (B, K)
        noise = torch.rand(present.shape, generator=generator, device=present.device)
        # Padding is given the highest noise, so only present patches are hidden.
        order = noise.masked_fill(~present, 2.0).argsort(dim=1)  # (B, K)
        rank = order.argsort(dim=1)  # (B, K)
        hidden = rank < (ratio * present.sum(dim=1, keepdim=True)).floor()  # (B, K)
        masked[name] = replace(tokens, visible=present & ~hidden)
    return masked
