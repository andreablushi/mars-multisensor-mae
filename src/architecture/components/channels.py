"""What a patch's channels are, which every part that reads one has to say."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from building.common.layout import Axis
from torch import Tensor, nn


def channel_axis(axes: Sequence[str]) -> int | None:
    """Return which axis of a patch its channels run along.

    Args:
        axes: What each axis of the instrument's values holds.

    Returns:
        at: The one axis that is not ground, or None where a patch is ground alone.
    """
    return next((at for at, holds in enumerate(axes) if holds != Axis.GROUND), None)


def channel_vectors(shape: Sequence[int], at: int | None, dim: int) -> nn.Parameter:
    """Return one learned vector per channel of an instrument, at one token width.

    Args:
        shape: The shape of one patch of the instrument.
        at: Which axis its channels run along, or None where a patch holds one.
        dim: The token width.

    Returns:
        channel: One vector per channel, drawn small. (C, D)
    """
    held = nn.Parameter(torch.zeros(shape[at] if at is not None else 1, dim))
    nn.init.normal_(held, std=0.02)
    return held


def by_channel(held: Tensor, at: int | None) -> Tensor:
    """Return one batch of patches as the ground samples of each of their channels.

    Args:
        held: One batch of patches, or of their validity. (B, K, *P)
        at: Which axis the channels run along, or None where a patch holds one.

    Returns:
        samples: The ground samples of each channel. (B, K, C, G)
    """
    if at is None:
        # Reshape single-channel patch to (B, K, 1, G) with single channel dimension
        return held.flatten(2).unsqueeze(2)  # (B, K, 1, G)
    # Move the channel axis forward and flatten the ground samples
    return held.movedim(2 + at, 2).flatten(3)  # (B, K, C, G)
