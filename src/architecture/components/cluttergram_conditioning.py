"""Telling a SHARAD token what its patch's cluttergram holds."""

from __future__ import annotations

from torch import Tensor, nn

# B = batch, K = patches, C = condition values, D = token channels.


class CluttergramConditioning(nn.Module):
    """Add each patch's cluttergram, mapped to the token width, to its token.

    Attributes:
        project: From a patch's cluttergram to the token width, started at zero so
            a conditioned model starts as an unconditioned one.
    """

    def __init__(self, width: int, dim: int) -> None:
        """Build the conditioning for one token width.

        Args:
            width: How many values each patch is conditioned on.
            dim: The token width.
        """
        super().__init__()
        self.project = nn.Linear(width, dim, bias=False)
        nn.init.zeros_(self.project.weight)

    def forward(self, tokens: Tensor, condition: Tensor) -> Tensor:
        """Return the tokens told what their patches are conditioned on.

        Args:
            tokens: One token per patch. (B, K, D)
            condition: What each patch is conditioned on, flattened. (B, K, C)

        Returns:
            tokens: The same tokens, conditioned. (B, K, D)
        """
        return tokens + self.project(condition)
