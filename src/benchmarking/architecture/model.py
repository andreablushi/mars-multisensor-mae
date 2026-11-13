"""The model a benchmark trains: a copy of the MAE under the attentive probe."""

from __future__ import annotations

from copy import deepcopy

from torch import Tensor, nn

from architecture.mae import CrossSensorMAE
from architecture.models.context import Context
from architecture.models.tokens import Tokens
from benchmarking.architecture.attentive_probe import AttentiveProbe
from configs.load import named_rng
from dataset.loader import TARGETS

PROTOCOLS = {"attentive-probe": False, "fine-tune": True}


class BenchmarkModel(nn.Module):
    """The MAE under the attentive probe, its encoders trained too when fine-tuned.

    Attributes:
        mae: The copy, frozen whole or with its encoders left to train.
        probe: The attentive probe, reading every instrument's tokens.
    """

    def __init__(
        self,
        mae: CrossSensorMAE,
        dim: int,
        heads: int,
        depth: int,
        finetuned: bool,
        seed: int,
    ) -> None:
        """Build the model over one MAE, left as it is.

        Args:
            mae: The MAE.
            dim: The token width the MAE hands its tokens at.
            heads: How many attention heads the probe runs.
            depth: How many cross-attention blocks the probe stacks.
            finetuned: Whether the encoders and the cross-sensor encoder are trained,
                or the MAE is left frozen whole.
            seed: What fixes the probe as drawn.
        """
        super().__init__()
        self.mae = deepcopy(mae).requires_grad_(False)
        self.mae.encoders.requires_grad_(finetuned)
        self.mae.crossencoder.requires_grad_(finetuned)
        with named_rng(seed, "benchmark/probe"):
            self.probe = AttentiveProbe(dim, heads, depth, len(TARGETS))

    def tokens(self, batch: dict[str, Tokens]) -> Context:
        """Return every instrument's tokens of one batch, as the probe reads them.

        Args:
            batch: Each instrument's patches over the batch, on the device.

        Returns:
            context: Every instrument's tokens side by side, without their CLS.
        """
        present = {name: tokens.present for name, tokens in batch.items()}
        _, contexts = self.mae.embed_tokens(batch, present)
        return Context.joined(
            [Context(*(part[:, 1:] for part in held)) for held in contexts.values()]
        )

    def forward(self, batch: dict[str, Tokens]) -> Tensor:
        """Return every point's prediction of each target over one batch.

        Args:
            batch: Each instrument's patches over the batch, on the device.

        Returns:
            prediction: One per point and target. (B, P * P, Z)
        """
        return self.probe(self.tokens(batch))
