"""One batch on the model device: its tokens read, then a head's logits."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from architecture.mae import CrossSensorMAE
from architecture.models.context import Context
from architecture.models.tokens import Tokens


def batch_logits(
    model: CrossSensorMAE,
    head: nn.Module,
    batch: dict[str, Tokens],
    device: torch.device,
) -> Tensor:
    """Return a head's logits over one batch, read from every token of each tile.

    Args:
        model: The model, on the device.
        head: The classifier or the segmenter, on the device.
        batch: Each instrument's patches over the batch, as the loader hands it.
        device: Where the model runs.

    Returns:
        logits: The head's logits, in float. (B, Z) or (B, P * P, Z)
    """
    batch = {name: tokens.to(device) for name, tokens in batch.items()}
    present = {name: tokens.present for name, tokens in batch.items()}
    with torch.autocast(device.type, dtype=torch.bfloat16):
        cls, contexts = model.embed_tokens(batch, present)
        context = Context.joined(
            [Context(*(part[:, 1:] for part in held)) for held in contexts.values()]
        )
        return head(cls, context).float()
