"""How alike tiles are: their tokens pooled into one vector, and the cosines between."""

from __future__ import annotations

import numpy as np
from torch import Tensor
from torch.nn import functional


def global_average_pooling(tokens: Tensor, present: Tensor) -> Tensor:
    """Return each row's present tokens summed, of unit length.

    Args:
        tokens: The tokens of every row. (B, K, D)
        present: Which of them hold something rather than padding. (B, K)

    Returns:
        pooled: One unit vector per row. (B, D)
    """
    return functional.normalize((tokens * present.unsqueeze(-1)).sum(dim=1), dim=-1)


def cosine_similarity(vectors: np.ndarray) -> np.ndarray:
    """Return the cosine between every pair of tiles' vectors.

    Args:
        vectors: One unit vector per tile. (T, D)

    Returns:
        similarities: The cosine of every pair. (T, T)
    """
    return vectors @ vectors.T
