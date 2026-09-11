"""Laying a latent space out on a plane, to look at what it did with the classes."""

from __future__ import annotations

import numpy as np
from torch import Tensor
from umap import UMAP


def projected_latent(latents: Tensor, seed: int) -> np.ndarray:
    """Return every latent laid out in two dimensions.

    Args:
        latents: One vector per feature, of unit length. (N, D)
        seed: The number that fixes the layout.

    Returns:
        placed: Where each of them sits on the plane, in that same order. The
            neighbourhoods it keeps are read by the same cosine the metrics
            are, so a class held together is a class drawn together. (N, 2)
    """
    return UMAP(n_components=2, metric="cosine", random_state=seed).fit_transform(
        latents.numpy()
    )  # (N, 2)
