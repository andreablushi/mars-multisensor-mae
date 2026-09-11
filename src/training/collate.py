"""Batching what one split hands back into the tokens the model is handed."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from architecture.tokens import Tokens


def collate(
    samples: list[tuple[dict[str, dict[str, np.ndarray]], str]],
) -> tuple[dict[str, Tokens], list[str]]:
    """Return one batch of every instrument's tokens, and the features' classes.

    Args:
        samples: What the dataset read of each feature of the batch.

    Returns:
        batch: Each instrument's patches padded to the most any feature of
            the batch holds, keyed as ODE names it.
        classes: The class of each feature, in the batch's order.
    """
    batch = {}
    for name in samples[0][0]:
        held = [sample[name] for sample, _ in samples]
        counts = torch.tensor([len(one["values"]) for one in held])  # (B,)
        slots = torch.arange(int(counts.max()))  # (K,)
        padded = {
            key: pad_sequence(
                [torch.as_tensor(one[key]) for one in held], batch_first=True
            )
            for key in held[0]
        }
        batch[name] = Tokens(**padded, present=slots.unsqueeze(0) < counts.unsqueeze(1))
    return batch, [feature_class for _, feature_class in samples]
