"""Telling a SHARAD token what its patch's cluttergram holds."""

from __future__ import annotations

from torch import Tensor, nn

# B = batch, K = patches, C = condition values, D = token channels.


class CluttergramConditioning(nn.Module):
    """Add each patch's cluttergram, mapped to the token width, to its token.

    Attributes:
        project: From a patch's cluttergram to the token width, started at zero so
            a conditioned model starts as an unconditioned one.
        scale: From a patch's cluttergram to a gain on each token channel, FiLM's
            scale, started at zero too, or None for a shift alone.
    """

    def __init__(self, width: int, dim: int, gain: bool) -> None:
        """Build the conditioning for one token width.

        Args:
            width: How many values each patch is conditioned on.
            dim: The token width.
            gain: Whether the cluttergram also scales each token, as FiLM does.
        """
        super().__init__()
        self.project = nn.Linear(width, dim, bias=False)
        nn.init.zeros_(self.project.weight)
        self.scale = nn.Linear(width, dim, bias=False) if gain else None
        if self.scale is not None:
            nn.init.zeros_(self.scale.weight)

    def forward(self, tokens: Tensor, condition: Tensor) -> Tensor:
        """Return the tokens told what their patches are conditioned on.

        Args:
            tokens: One token per patch. (B, K, D)
            condition: What each patch is conditioned on, flattened. (B, K, C)

        Returns:
            tokens: The same tokens, conditioned. (B, K, D)
        """
        if self.scale is not None:
            tokens = tokens * (1 + self.scale(condition))
        return tokens + self.project(condition)
