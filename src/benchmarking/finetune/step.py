"""One batch on the model device: its tokens read once, then every head's answer."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from analysis.ground_truth.swim.sample import GRID_SPACING_KM
from common.maths.physics import METRES_PER_KM
from torch import Tensor, nn

from architecture.mae import CrossSensorMAE
from architecture.models.context import Context
from architecture.models.tokens import Tokens
from benchmarking.architecture.segmenter import grid_points


def batch_consistency(
    model: CrossSensorMAE,
    heads: Mapping[str, nn.Module],
    batch: dict[str, Tokens],
    device: torch.device,
) -> tuple[dict[str, Tensor], Tensor]:
    """Return every head's consistency over one batch, and which points it read.

    Args:
        model: The model, on the device.
        heads: The classifier and the segmenter, on the device, keyed by task.
        batch: Each instrument's patches over the batch, as the loader hands it.
        device: Where the model runs.

    Returns:
        consistency: Each head's consistency, in float, keyed by task. (B, Z) or
            (B, P * P, Z)
        covered: Which grid points have a token in the cell around them. (B, P * P)
    """
    batch = {name: tokens.to(device) for name, tokens in batch.items()}
    present = {name: tokens.present for name, tokens in batch.items()}
    with torch.autocast(device.type, dtype=torch.bfloat16):
        cls, contexts = model.embed_tokens(batch, present)
        context = Context.joined(
            [Context(*(part[:, 1:] for part in held)) for held in contexts.values()]
        )
        consistency = {task: head(cls, context).float() for task, head in heads.items()}
    points = grid_points().to(device)
    apart = (context.position[:, :, None, :2] - points[:, :2]).abs().amax(-1)
    near = apart <= GRID_SPACING_KM * METRES_PER_KM / 2  # (B, S, P * P)
    return consistency, (near & context.readable[..., None].bool()).any(1)
