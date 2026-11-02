"""Rebuilding one tile's observations, and where each CMR looked on the others."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from building.configs import sharad
from building.metadata.observation import ObservationMetadata
from torch import Tensor
from torch.nn import functional

from architecture.components.decoder import Decoder
from architecture.mae import CrossSensorMAE
from architecture.models.context import Context
from architecture.models.tokens import Tokens
from configs.load import mask_generators
from dataset.loader import padded_tokens
from dataset.patches import cut_tile_patches, patch_arrays
from dataset.store import DatasetBuild
from training.step import drawn_masks


def reconstruction_and_attention(
    model: CrossSensorMAE,
    build: DatasetBuild,
    rows: Mapping[str, Sequence[ObservationMetadata]],
    shown: Mapping[str, str],
    sizes: Mapping[str, Mapping[str, int]],
    shapes: Mapping[str, tuple[int, ...]],
    mask_ratio: float,
    ranging: Sequence[str],
    seed: int,
    device: torch.device,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, dict[str, np.ndarray]]],
    dict[str, dict[str, dict[str, np.ndarray]]],
    dict[str, dict[str, np.ndarray]],
]:
    """Return each instrument's shown observation rebuilt, and rebuilt from others.

    Args:
        model: The model, on the device.
        build: The build the tile is read from.
        rows: The tile's index rows of each instrument.
        shown: The observation of each instrument shown, by its identifier.
        sizes: How far a patch of each sensor runs along each axis it is cut on.
        shapes: The shape of one patch of each instrument as the model reads it.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        ranging: The two ranging instruments, each also rebuilt from the other alone.
        seed: What fixes the masks.
        device: Where the model runs.

    Returns:
        reconstruction: Each instrument's shown observation as drawn_arrays names
            it, the "hidden" patches, and those rebuilt under "umr" and "cmr".
        attention: For each instrument rebuilt from every other whole, keyed by it,
            what cmr_attention hands back.
        ranged: The same for each ranging instrument, rebuilt from the other alone.
        tokens: Every observation of each ground instrument and the sounder's shown
            pass, alike, with their tokens in the shared space, every patch read,
            under "token".
    """
    alone, every = {}, {}
    for name in sizes:
        (row,) = (one for one in rows[name] if one.identifier == shown[name])
        alone[name] = drawn_arrays(build, name, [row], sizes[name], shapes[name])
        # Every observation least measured first, so the most measured is laid last
        held = sorted(rows[name], key=lambda one: one.valid_count)
        every[name] = drawn_arrays(build, name, held, sizes[name], shapes[name])
    model.eval()
    # The query's shown observation hidden whole, every source's all read whole
    passes = {
        query: cmr_attention(model, every | {query: alone[query]}, query, device)
        for query in sizes
    }
    ranged = {
        query: cmr_attention(
            model,
            {
                name: (alone if name == query else every)[name]
                for name in sizes
                if name in (query, other)
            },
            query,
            device,
        )[0]
        for query, other in zip(ranging, ranging[::-1], strict=True)
    }
    return (
        masked_reconstruction(model, alone, mask_ratio, seed, device),
        {query: attended for query, (attended, _) in passes.items()},
        ranged,
        # A sounder's passes cross, so its query pass reads one alone whole
        passes[sharad.LAYOUT.instrument][1],
    )


def drawn_arrays(
    build: DatasetBuild,
    name: str,
    rows: Sequence[ObservationMetadata],
    size: Mapping[str, int],
    shape: tuple[int, ...],
) -> dict[str, np.ndarray]:
    """Return some observations of one instrument as the arrays drawn.

    Args:
        build: The build they are read from.
        name: The instrument, as ODE names it.
        rows: The observations' index rows, in the order they are laid.
        size: How far a patch runs along each axis it is cut on.
        shape: The shape of one patch as the model reads it.

    Returns:
        arrays: Their patches as patch_arrays names them, and "cell", the "clip",
            "mean" and "std" of the z-scores, "spacing", "observation" and
            "identifier".
    """
    read = [cut_tile_patches({name: [row]}, build, {name: size})[name] for row in rows]
    patches = [patch for held in read for patch in held]
    counts = [len(held) for held in read]
    stats = build.read_stats()[name]
    return patch_arrays(patches, name, shape, rows[0].axes, build.acquisition[name]) | {
        "cell": np.array([one.cell for one in patches]),
        "clip": np.array(build.clips[name], dtype=float),
        "mean": np.array(stats["mean"], dtype=float),
        "std": np.array(stats["std"], dtype=float),
        "spacing": np.repeat([row.sample_spacing_m for row in rows], counts, axis=0),
        "observation": np.repeat(np.arange(len(rows)), counts),
        "identifier": np.array([row.identifier for row in rows]),
    }


def batch_on_device(
    sample: Mapping[str, Mapping[str, np.ndarray]], device: torch.device
) -> dict[str, Tokens]:
    """Return one tile's arrays as a batch of one, on the device.

    Args:
        sample: Each instrument's arrays, as drawn_arrays names them.
        device: Where the model runs.

    Returns:
        batch: Each instrument's patches, a batch of one.
    """
    return {name: one.to(device) for name, one in padded_tokens([sample]).items()}


def masked_reconstruction(
    model: CrossSensorMAE,
    sample: Mapping[str, Mapping[str, np.ndarray]],
    mask_ratio: float,
    seed: int,
    device: torch.device,
) -> dict[str, dict[str, np.ndarray]]:
    """Return one tile's arrays, a share of their patches hidden and rebuilt.

    Args:
        model: The model, in evaluation on the device.
        sample: Each instrument's arrays, as drawn_arrays names them.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        seed: What fixes the masks.
        device: Where the model runs.

    Returns:
        reconstruction: Each instrument's arrays, the "hidden" patches, and those
            rebuilt under "umr" and "cmr".
    """
    batch = batch_on_device(sample, device)
    visible, hidden = drawn_masks(
        batch, mask_ratio, mask_generators(batch, seed, device)
    )
    with torch.no_grad():
        predictions, _ = model(batch, visible, hidden, "GAP")
    return {
        name: one
        | {"hidden": hidden[name][0].cpu().numpy()}
        | {term: predictions[term][name][0].cpu().numpy() for term in ("umr", "cmr")}
        for name, one in sample.items()
    }


def cmr_attention(
    model: CrossSensorMAE,
    sample: Mapping[str, Mapping[str, np.ndarray]],
    query: str,
    device: torch.device,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    """Return the query rebuilt from every other instrument of a tile, and their tokens.

    Args:
        model: The model, in evaluation on the device.
        sample: Each instrument's arrays, as drawn_arrays names them, every patch
            read.
        query: The instrument rebuilt, hidden whole.
        device: Where the model runs.

    Returns:
        attended: The query's arrays with the rebuilt patches under "cmr", and every
            other's with their tokens' rolled out attention relative to uniform
            under "attention", the share all their tokens drew under "share" and
            their CLS under "cls".
        tokens: Every instrument's arrays with its tokens in the shared space under
            "token".
    """
    batch = batch_on_device(sample, device)
    present = {name: one.present for name, one in batch.items()}
    with torch.no_grad():
        _, contexts = model.embed_tokens(batch, present)
    sources = [name for name in sample if name != query]
    # CMR reads the others without the CLS of every instrument
    context = Context.joined([contexts[name] for name in sources])
    cmr, drawn = rolled_attention(
        model.decoders[query],
        context,
        batch[query].position,
        batch[query].acquisition,
        present[query],
    )
    readable = int(context.readable.sum())
    # Each instrument's CLS, then its tokens
    lengths = [contexts[name].tokens.shape[1] for name in sources]
    split = np.split(drawn.cpu().numpy(), np.cumsum(lengths)[:-1])
    attended = {query: sample[query] | {"cmr": cmr[0].cpu().numpy()}} | {
        name: sample[name]
        | {"attention": taken[1:] * readable, "share": taken.sum(), "cls": taken[0]}
        for name, taken in zip(sources, split, strict=True)
    }
    tokens = {
        name: one | {"token": contexts[name].tokens[0, 1:].cpu().numpy()}
        for name, one in sample.items()
    }
    return attended, tokens


def rolled_attention(
    decoder: Decoder,
    context: Context,
    position: Tensor,
    acquisition: Tensor,
    hidden: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the patches a decoder reads from one context, and what each token drew.

    Args:
        decoder: The decoder, its model in evaluation.
        context: The tokens it reads, a batch of one.
        position: Each asked patch's centre. (1, K, 2 or 3)
        acquisition: Each asked patch's acquisition angles, in degrees. (1, K, A)
        hidden: Which of them to predict. (1, K)

    Returns:
        prediction: One patch per slot, meaningful where hidden. (1, K, *P)
        drawn: The share of the asked patches' attention each context token drew,
            rolled out over the blocks and averaged over the patches. (S,)
    """
    # Record each block's attention while the pass runs
    weights = []
    hooks = [
        block.register_forward_hook(
            lambda block, args, _: weights.append(block.attention_weights(*args))
        )
        for block in decoder.blocks.blocks
    ]
    # The pass itself, the asked patches reading the context
    with torch.no_grad():
        prediction = decoder.reconstructed_patches(
            *context, position, acquisition, hidden
        )
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
