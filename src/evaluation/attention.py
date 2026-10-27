"""Where a decoder looks: its attention rolled out over its blocks."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional

from architecture.components.decoder import Decoder
from architecture.components.transformer import Block
from architecture.tokens import Context

# K = asked patches, S = context tokens, N = all tokens, D = token channels.


def rolled_attention(
    decoder: Decoder, context: Context, position: Tensor, hidden: Tensor
) -> tuple[Tensor, Tensor]:
    """Return the patches a decoder reads from one context, and what each token drew.

    Args:
        decoder: The decoder, its model in evaluation.
        context: The tokens it reads, a batch of one.
        position: Each asked patch's centre. (1, K, 2 or 3)
        hidden: Which of them to predict. (1, K)

    Returns:
        prediction: One patch per slot, meaningful where hidden. (1, K, *P)
        drawn: The share of the asked patches' attention each context token drew,
            rolled out over the blocks and averaged over the patches. (S,)
    """

    def block_attention(
        block: Block, keys: Tensor, mask: Tensor, placed: Tensor, fixed: int
    ) -> Tensor:
        """Return what each query of one block attends, its heads averaged."""
        query, key, _ = (
            block.qkv(block.attend_norm(keys))
            .unflatten(-1, (3, block.heads, -1))
            .permute(2, 0, 3, 1, 4)
        )  # (B, H, N, D / H) each
        scores = block.rotate(query[..., fixed:, :], placed[:, fixed:]) @ block.rotate(
            key, placed
        ).transpose(-2, -1)
        scores = scores / math.sqrt(query.shape[-1])
        return scores.masked_fill(~mask, -torch.inf).softmax(-1).mean(1)

    # Record each block's attention while the pass runs
    weights = []
    hooks = [
        block.register_forward_hook(
            lambda block, args, _: weights.append(block_attention(block, *args))
        )
        for block in decoder.blocks.blocks
    ]
    # The pass itself, the asked patches reading the context
    with torch.no_grad():
        prediction = decoder.reconstructed_patches(*context, position, hidden)
    for hook in hooks:
        hook.remove()
    # Roll out: half of each block passes on what the queries it read had gathered
    count, asked = context.tokens.shape[1], prediction.shape[1]
    device = prediction.device
    itself = functional.pad(torch.eye(asked, device=device), (count, 0))  # (K, S + K)
    rolled = torch.zeros(asked, count, device=device)  # (K, S)
    for attended in weights:
        mixed = (attended[0] + itself) / 2  # (K, S + K)
        rolled = mixed[:, :count] + mixed[:, count:] @ rolled
    # Average over the asked patches, as a share of all they drew
    drawn = rolled[hidden[0]].mean(0)
    return prediction, drawn / drawn.sum()
