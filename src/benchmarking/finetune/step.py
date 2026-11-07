"""One batch on the model device: its tokens read, then the head's answer."""

from __future__ import annotations

import torch
from analysis.ground_truth.swim.sample import GRID_SPACING_KM
from common.maths.physics import METRES_PER_KM
from torch import Tensor

from architecture.mae import CrossSensorMAE
from architecture.models.context import Context
from architecture.models.tokens import Tokens
from benchmarking.architecture.segmenter import Segmenter


def batch_consistency(
    model: CrossSensorMAE,
    head: Segmenter,
    batch: dict[str, Tokens],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Return the head's consistency over one batch, and which points it read.

    Args:
        model: The model, on the device.
        head: The map head, on the device.
        batch: Each instrument's patches over the batch, as the loader hands it.
        device: Where the model runs.

    Returns:
        consistency: The consistency of every point, in float. (B, P * P, Z)
        covered: Which grid points have a token in the cell around them. (B, P * P)
    """
    batch = {name: tokens.to(device) for name, tokens in batch.items()}
    present = {name: tokens.present for name, tokens in batch.items()}
    with torch.autocast(device.type, dtype=torch.bfloat16):
        _, contexts = model.embed_tokens(batch, present)
        context = Context.joined(
            [Context(*(part[:, 1:] for part in held)) for held in contexts.values()]
        )
        consistency = head(context).float()
    apart = (context.position[:, :, None, :2] - head.points[:, :2]).abs().amax(-1)
    near = apart <= GRID_SPACING_KM * METRES_PER_KM / 2  # (B, S, P * P)
    return consistency, (near & context.readable[..., None].bool()).any(1)
