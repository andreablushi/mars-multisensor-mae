"""Where one instrument's CMR looks: its decoder's attention over the others' tokens."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from building.metadata.observation import ObservationMetadata
from torch import Tensor
from torch.nn import functional

from architecture.components.transformer import Block
from architecture.mae import CrossSensorMAE
from architecture.tokens import Context
from dataset.patches import patch_arrays, read_tile_patches
from dataset.store import DatasetBuild
from evaluation.reconstruct import drawn_stats, shown_observation, tile_batch

# K = query patches, S = context tokens, N = all tokens, D = token channels.


def block_attention(
    block: Block, keys: Tensor, mask: Tensor, position: Tensor, fixed: int
) -> Tensor:
    """Return what each query of one block attends, its heads averaged.

    Args:
        block: The block, called on the same keys.
        keys: Every token, those past the fixed also asking as queries. (B, N, D)
        mask: Which key each query may attend to. (B, 1, N - fixed, N)
        position: Each token's patch centre. (B, N, 2 or 3)
        fixed: How many leading tokens are keys alone, never queries.

    Returns:
        attention: Each query's weights over every key. (B, N - fixed, N)
    """
    query, key, _ = (
        block.qkv(block.attend_norm(keys))
        .unflatten(-1, (3, block.heads, -1))
        .permute(2, 0, 3, 1, 4)
    )  # (B, H, N, D / H) each
    scores = block.rotate(query[..., fixed:, :], position[:, fixed:]) @ block.rotate(
        key, position
    ).transpose(-2, -1)
    scores = scores / math.sqrt(query.shape[-1])
    return scores.masked_fill(~mask, -torch.inf).softmax(-1).mean(1)


def attend(
    model: CrossSensorMAE,
    build: DatasetBuild,
    rows: Mapping[str, Sequence[ObservationMetadata]],
    axes: Mapping[str, tuple[str, ...]],
    sizes: Mapping[str, Mapping[str, int]],
    shapes: Mapping[str, tuple[int, ...]],
    query: str,
    device: torch.device,
) -> dict[str, dict[str, np.ndarray]]:
    """Return one instrument rebuilt from every other whole, and what it attended.

    Args:
        model: The model, on the device.
        build: The build the tile is read from.
        rows: The tile's index rows of each instrument.
        axes: What each axis of each instrument's values holds.
        sizes: How far a patch of each sensor runs along each axis it is cut on.
        shapes: The shape of one patch of each instrument as the model reads it.
        query: The instrument rebuilt, whose shown observation is hidden whole.
        device: Where the model runs.

    Returns:
        attention: Each instrument's patches under "values", "measured", "cell",
            "position", "spacing", "observation" and "identifier", with the "clip",
            "mean" and "std" of their z-scores; the query's rebuilt under "cmr", every
            other's tokens' rolled out attention relative to uniform under
            "attention", and the share all its tokens drew under "share" and its CLS
            under "cls".
    """
    # The query's shown observation, every other's all, the most measured last
    observations = {
        name: [shown_observation(rows[name], axes[name])]
        if name == query
        else sorted(rows[name], key=lambda one: one.valid_count)
        for name in sizes
    }
    sample = {}
    for name, held in observations.items():
        read = [
            read_tile_patches({name: [row]}, build, {name: sizes[name]})[name]
            for row in held
        ]
        counts = [len(one) for one in read]
        patches = [patch for one in read for patch in one]
        sample[name] = (
            patch_arrays(
                patches,
                shapes[name],
                axes[name],
                build.condition_width(name, shapes[name]),
            )
            | {
                "cell": np.array([one.cell for one in patches]),
                "spacing": np.repeat(
                    [row.sample_spacing_m for row in held], counts, axis=0
                ),
                "observation": np.repeat(np.arange(len(held)), counts),
                "identifier": np.array([row.identifier for row in held]),
            }
            | drawn_stats(build, name)
        )
    batch = tile_batch(sample, device)
    present = {name: one.present for name, one in batch.items()}
    others = [name for name in batch if name != query]
    decoder = model.decoders[query]
    weights = []
    hooks = [
        block.register_forward_hook(
            lambda block, args, _: weights.append(block_attention(block, *args))
        )
        for block in decoder.blocks.blocks
    ]
    model.eval()
    with torch.no_grad():
        cls, tokens = model.shared_tokens(batch, present)
        sources = model.context(cls, tokens, batch, present, others)
        # CMR reads the others without the CLS of every instrument
        context = Context(*(one[:, 1:] for one in sources))
        cmr = decoder.reconstructed_patches(
            *context,
            batch[query].position,
            present[query],
        )  # (1, K, *P)
    for hook in hooks:
        hook.remove()
    # Rollout: half of each block passes on what the queries it read had gathered
    count, asked = context.tokens.shape[1], cmr.shape[1]
    itself = functional.pad(torch.eye(asked, device=device), (count, 0))  # (K, S + K)
    rolled = torch.zeros(asked, count, device=device)  # (K, S)
    for attended in weights:
        mixed = (attended[0] + itself) / 2  # (K, S + K)
        rolled = mixed[:, :count] + mixed[:, count:] @ rolled
    drawn = rolled[present[query][0]].mean(0)
    drawn = (drawn / drawn.sum()).cpu().numpy()  # (S,)
    readable = int(context.readable.sum())
    attention = {query: sample[query] | {"cmr": cmr[0].cpu().numpy()}}
    start = 0
    for name in others:
        held = drawn[start : start + 1 + batch[name].values.shape[1]]
        start += held.size
        attention[name] = sample[name] | {
            "attention": held[1:] * readable,
            "share": held.sum(),
            "cls": held[0],
        }
    return attention
