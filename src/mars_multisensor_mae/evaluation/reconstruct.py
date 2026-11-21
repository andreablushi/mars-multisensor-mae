"""Rebuilding one tile's observations, and where each CMR looked on the others."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from mars_multisensor_dataset.building.metadata.observation import ObservationMetadata
from torch import Tensor
from torch.nn import functional
from wandb.sdk.wandb_run import Run

from mars_multisensor_mae.architecture.components.decoder import Decoder
from mars_multisensor_mae.architecture.mae import CrossSensorMAE
from mars_multisensor_mae.architecture.models.context import Context
from mars_multisensor_mae.architecture.models.tokens import Tokens
from mars_multisensor_mae.configs.load import mask_generators
from mars_multisensor_mae.configs.schema import DrawnTile
from mars_multisensor_mae.dataset.loader import padded_tokens
from mars_multisensor_mae.dataset.models.patch import PatchLayout
from mars_multisensor_mae.dataset.patches import drawn_arrays
from mars_multisensor_mae.dataset.store import DatasetBuild
from mars_multisensor_mae.evaluation.results import write_reconstructions
from mars_multisensor_mae.logs.evaluation import log_reconstruction
from mars_multisensor_mae.training.step import masking_random_correspondence

NEAR_M = 3000.0


def evaluate_reconstruction(
    model: CrossSensorMAE,
    build: DatasetBuild,
    tiles: Mapping[str, DrawnTile],
    layout: PatchLayout,
    mask_ratio: float,
    ranging: Sequence[str],
    seed: int,
    device: torch.device,
    run: Run,
    run_name: str,
) -> None:
    """Write and log a few tiles rebuilt, where CMR looked, and each ranging pair.

    Args:
        model: The model, on the device.
        build: The build the tiles are read from.
        tiles: The tiles drawn as reconstructions and attention maps, by what they
            show, with each instrument's observation shown.
        layout: How each instrument is cut into patches, and the shape the model
            reads one in.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        ranging: The two ranging instruments, each also rebuilt from the other alone.
        seed: What fixes the masks.
        device: Where the model runs.
        run: The tracked run.
        run_name: What the run is called, which names its results and artifact.
    """
    by_tile = build.read_observation_metadata()
    reconstructions, rangings = {}, {}
    for place, drawn in tiles.items():
        by_query, ranged = reconstruction_and_attention(
            model,
            build,
            by_tile[drawn.tile],
            drawn.shown,
            layout,
            mask_ratio,
            ranging,
            seed,
            device,
        )
        reconstructions |= {(place, query): one for query, one in by_query.items()}
        rangings |= {(place, query): one for query, one in ranged.items()}
    log_reconstruction(
        run,
        run_name,
        reconstructions,
        rangings,
        write_reconstructions(run_name, reconstructions, rangings),
    )


def reconstruction_and_attention(
    model: CrossSensorMAE,
    build: DatasetBuild,
    rows: Mapping[str, Sequence[ObservationMetadata]],
    shown: Mapping[str, str],
    layout: PatchLayout,
    mask_ratio: float,
    ranging: Sequence[str],
    seed: int,
    device: torch.device,
) -> tuple[
    dict[str, dict[str, dict[str, np.ndarray]]],
    dict[str, dict[str, dict[str, np.ndarray]]],
]:
    """Return each instrument's shown observation rebuilt from the others, and masked.

    Args:
        model: The model, on the device.
        build: The build the tile is read from.
        rows: The tile's index rows of each instrument.
        shown: The observation of each instrument shown, by its identifier.
        layout: How each instrument is cut into patches, and the shape the model
            reads one in.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        ranging: The two ranging instruments, each also rebuilt from the other alone.
        seed: What fixes the masks.
        device: Where the model runs.

    Returns:
        attention: Per instrument rebuilt from every other whole, what cmr_attention
            hands back, the query's arrays with masked_reconstruction's too.
        ranged: The same for each ranging instrument from the other alone, unmasked.
    """
    alone, every = {}, {}
    for name in layout.sizes:
        (row,) = (one for one in rows[name] if one.identifier == shown[name])
        alone[name] = drawn_arrays(
            build, name, [row], layout.sizes[name], layout.shapes[name]
        )
        # Every observation least measured first, so the most measured is laid last
        held = sorted(rows[name], key=lambda one: one.valid_count)
        every[name] = drawn_arrays(
            build, name, held, layout.sizes[name], layout.shapes[name]
        )
    masked = masked_reconstruction(model, alone, mask_ratio, seed, device)
    # The query's shown observation hidden whole, every source's all read whole
    attention = {
        query: cmr_attention(
            model, every | {query: alone[query] | masked[query]}, query, device
        )
        for query in layout.sizes
    }
    ranged = {
        query: cmr_attention(
            model,
            {
                name: (alone if name == query else every)[name]
                for name in layout.sizes
                if name in (query, other)
            },
            query,
            device,
        )
        for query, other in zip(ranging, ranging[::-1], strict=True)
    }
    return attention, ranged


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
        model: The model, on the device.
        sample: Each instrument's arrays, as drawn_arrays names them.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        seed: What fixes the masks.
        device: Where the model runs.

    Returns:
        masked: Each instrument's "hidden" patches, and those rebuilt under "umr".
    """
    batch = batch_on_device(sample, device)
    visible, hidden = masking_random_correspondence(
        batch, mask_ratio, mask_generators(batch, seed, device)
    )
    with torch.no_grad():
        predictions, _ = model(batch, visible, hidden, "GAP")
    return {
        name: {
            "hidden": hidden[name][0].cpu().numpy(),
            "umr": predictions["umr"][name][0].cpu().numpy(),
        }
        for name in sample
    }


def cmr_attention(
    model: CrossSensorMAE,
    sample: Mapping[str, Mapping[str, np.ndarray]],
    query: str,
    device: torch.device,
) -> dict[str, dict[str, np.ndarray]]:
    """Return the query rebuilt from every other instrument, and what each token drew.

    Args:
        model: The model, on the device.
        sample: Each instrument's arrays, as drawn_arrays names them, every patch
            read.
        query: The instrument rebuilt, hidden whole.
        device: Where the model runs.

    Returns:
        attended: The query's arrays plus "cmr", every other's plus rolled_attention's
            drawn and near over uniform as "overall" and "attention", "share", "cls".
    """
    batch = batch_on_device(sample, device)
    present = {name: one.present for name, one in batch.items()}
    with torch.no_grad():
        encoded = model.embed_tokens(batch, present)
    sources = [name for name in sample if name != query]
    # CMR reads the others without the CLS of every instrument
    context = encoded.cross_modal_reconstruction_context(query)
    cmr, drawn, near = rolled_attention(
        model.decoders[query],
        context,
        batch[query].position,
        batch[query].acquisition,
        present[query],
    )
    readable = int(context.readable.sum())
    # Each instrument's CLS, then its tokens
    lengths = [encoded.instruments[name].tokens.shape[1] for name in sources]
    ends = np.cumsum(lengths)[:-1]
    split = zip(
        sources,
        np.split(drawn.cpu().numpy(), ends),
        np.split(near.cpu().numpy(), ends),
        strict=True,
    )
    return {query: sample[query] | {"cmr": cmr[0].cpu().numpy()}} | {
        name: sample[name]
        | {
            "overall": taken[1:] * readable,
            "attention": local[1:] * readable,
            "share": taken.sum(),
            "cls": taken[0],
        }
        for name, taken, local in split
    }


def rolled_attention(
    decoder: Decoder,
    context: Context,
    position: Tensor,
    acquisition: Tensor,
    hidden: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return the patches a decoder reads from one context, and what each token drew.

    Args:
        decoder: The decoder.
        context: The tokens it reads, a batch of one.
        position: Each asked patch's centre. (1, K, 2 or 3)
        acquisition: Each asked patch's acquisition angles, in degrees. (1, K, A)
        hidden: Which of them to predict. (1, K)

    Returns:
        prediction: One patch per slot, meaningful where hidden. (1, K, *P)
        drawn: The share of the asked patches' attention each context token drew,
            rolled out over the blocks and averaged over the patches. (S,)
        near: The same share averaged over the asked patches within NEAR_M of the
            token on the ground alone, NaN where none is. (S,)
    """
    # Record each block's attention while the pass runs
    weights = []
    hooks = [
        block.register_forward_hook(
            lambda block, args, _: weights.append(block.head_averaged_attention(*args))
        )
        for block in decoder.blocks.blocks
    ]
    # The pass itself, the asked patches reading the context
    with torch.no_grad():
        prediction = decoder(context, position, acquisition, hidden)
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
    rolled = rolled[hidden[0]]  # (K', S)
    drawn = rolled.mean(0)
    shares = rolled / rolled.sum(1, keepdim=True)
    close = (
        torch.cdist(position[0, hidden[0], :2], context.position[0, :, :2]) < NEAR_M
    )  # (K', S)
    near = (shares * close).sum(0) / close.sum(0)
    return prediction, drawn / drawn.sum(), near
