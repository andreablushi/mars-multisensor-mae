"""Fine-tuning a model's encoders through low-rank updates, as LoRA adapts them."""

from __future__ import annotations

import math
from copy import deepcopy

import torch
from torch import Tensor, nn
from torch.nn import functional

from architecture.components.transformer import Block
from architecture.mae import CrossSensorMAE


class LoRALinear(nn.Module):
    """A frozen linear layer beside a trained low-rank update of its weight.

    Attributes:
        linear: The frozen layer.
        down: From a token to the update's rank, as Kaiming draws it. (R, I)
        up: From the rank back to the layer's width, zero so it starts as the
            frozen layer. (O, R)
        scale: What the update is multiplied by, alpha over the rank.
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float) -> None:
        """Build the update beside one layer.

        Args:
            linear: The frozen layer.
            rank: The update's rank.
            alpha: What the update is scaled by before it is divided by the rank.
        """
        super().__init__()
        self.linear = linear
        self.down = nn.Parameter(torch.empty(rank, linear.in_features))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
        self.up = nn.Parameter(torch.zeros(linear.out_features, rank))
        self.scale = alpha / rank

    def forward(self, tokens: Tensor) -> Tensor:
        """Return the frozen layer's output plus the update's.

        Args:
            tokens: The tokens. (..., I)

        Returns:
            out: The tokens through the layer. (..., O)
        """
        update = functional.linear(functional.linear(tokens, self.down), self.up)
        return self.linear(tokens) + self.scale * update


def adapted_model(model: CrossSensorMAE, rank: int, alpha: float) -> CrossSensorMAE:
    """Return a frozen copy of the model, LoRA beside each encoder attention layer.

    Args:
        model: The model, left as it is.
        rank: Each update's rank.
        alpha: What each update is scaled by before it is divided by the rank.

    Returns:
        adapted: The copy, only its updates training.
    """
    adapted = deepcopy(model).requires_grad_(False)
    for part in (adapted.encoders, adapted.crossencoder):
        for block in part.modules():
            if isinstance(block, Block):
                block.qkv = LoRALinear(block.qkv, rank, alpha)
                block.out = LoRALinear(block.out, rank, alpha)
    return adapted
