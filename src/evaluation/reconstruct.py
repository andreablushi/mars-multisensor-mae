"""Rebuilding one tile's observations, and where each CMR looked on the others."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from building.common.layout import Axis
from building.metadata.observation import ObservationMetadata

from architecture.mae import CrossSensorMAE
from architecture.tokens import GAP, Context, Tokens, padded_batch
from dataset.patches import cut_tile_patches, patch_arrays
from dataset.store import DatasetBuild
from evaluation.attention import rolled_attention
from training.step import drawn_masks, mask_generators

SOUNDER_INCIDENCE_DEG = 140.0


def reconstruction_and_attention(
    model: CrossSensorMAE,
    build: DatasetBuild,
    rows: Mapping[str, Sequence[ObservationMetadata]],
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
        sizes: How far a patch of each sensor runs along each axis it is cut on.
        shapes: The shape of one patch of each instrument as the model reads it.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        ranging: The two ranging instruments, each also rebuilt from the other alone.
        seed: What fixes the masks.
        device: Where the model runs.

    Returns:
        reconstruction: Each instrument's shown observation under "values",
            "measured" and "cell", the "clip", "mean" and "std" of its z-scores, its
            "spacing", "observation" and "identifier", the "hidden" patches, and
            those rebuilt under "umr" and "cmr".
        attention: For each instrument rebuilt from every other whole, keyed by it,
            its shown observation with the rebuilt one under "cmr", and every
            other's observations with their tokens' rolled out attention relative to
            uniform under "attention", the share all their tokens drew under
            "share" and their CLS under "cls".
        ranged: The same for each ranging instrument, rebuilt from the other alone.
        tokens: Every observation of each ground instrument and the sounder's shown
            pass, alike, with their tokens in the shared space, every patch read,
            under "token".
    """
    # The most measured observation of each instrument is shown, a sounder's pass
    # nearest SOUNDER_INCIDENCE_DEG, and every observation is read once, least
    # measured first
    shown, held, read, whole = {}, {}, {}, {}
    for name in sizes:
        sounder = Axis.DELAY in rows[name][0].axes
        if sounder:
            shown[name] = min(
                rows[name],
                key=lambda one: abs(
                    one.acquisition.incidence_deg - SOUNDER_INCIDENCE_DEG
                ),
            )
        else:
            shown[name] = max(rows[name], key=lambda one: one.valid_count)
        held[name] = sorted(rows[name], key=lambda one: one.valid_count)
        # A sounder's passes cross, so one alone is drawn whole
        whole[name] = [shown[name]] if sounder else held[name]
        for row in held[name]:
            read[row.path] = cut_tile_patches(
                {name: [row]}, build, {name: sizes[name]}
            )[name]

    def arrays(name: str, observations: Sequence[ObservationMetadata]) -> dict:
        """Return some observations of one instrument as the arrays drawn."""
        patches = [patch for row in observations for patch in read[row.path]]
        counts = [len(read[row.path]) for row in observations]
        return (
            patch_arrays(
                patches,
                shapes[name],
                observations[0].axes,
                build.condition_width(name, shapes[name]),
                len(build.acquired(name)),
            )
            | {"cell": np.array([one.cell for one in patches])}
            | {"clip": np.array(build.clips[name], dtype=float)}
            | {
                key: np.array(build.read_stats()[name][key], dtype=float)
                for key in ("mean", "std")
            }
            | {
                "spacing": np.repeat(
                    [row.sample_spacing_m for row in observations], counts, axis=0
                ),
                "observation": np.repeat(np.arange(len(observations)), counts),
                "identifier": np.array([row.identifier for row in observations]),
            }
        )

    def on_device(sample: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, Tokens]:
        """Return one tile's arrays as a batch of one, on the device."""
        batch, _ = padded_batch([(sample, "")])
        return {
            name: Tokens(*(one.to(device) for one in tokens))
            for name, tokens in batch.items()
        }

    model.eval()
    # Every shown observation, a share of its patches hidden and rebuilt
    sample = {name: arrays(name, [shown[name]]) for name in sizes}
    batch = on_device(sample)
    visible, hidden = drawn_masks(
        batch, mask_ratio, mask_generators(batch, seed, device)
    )
    with torch.no_grad():
        predictions, _ = model(batch, visible, hidden, GAP)
    reconstruction = {
        name: one
        | {"hidden": hidden[name][0].cpu().numpy()}
        | {term: predictions[term][name][0].cpu().numpy() for term in ("umr", "cmr")}
        for name, one in sample.items()
    }
    # Every observation drawn whole, every patch read
    sample = {name: arrays(name, kept) for name, kept in whole.items()}
    batch = on_device(sample)
    with torch.no_grad():
        _, tokens = model.shared_tokens(
            batch, {name: one.present for name, one in batch.items()}
        )
    tokens = {
        name: one | {"token": tokens[name][0, 1:].cpu().numpy()}
        for name, one in sample.items()
    }

    def cmr_attention(query: str, sources: Sequence[str]) -> dict:
        """Return the query rebuilt from some others whole, and where it looked."""
        # The query's shown observation hidden whole, every source's all read whole
        observations = {
            name: [shown[name]] if name == query else held[name]
            for name in sizes
            if name == query or name in sources
        }
        sample = {name: arrays(name, kept) for name, kept in observations.items()}
        batch = on_device(sample)
        present = {name: one.present for name, one in batch.items()}
        with torch.no_grad():
            cls, tokens = model.shared_tokens(batch, present)
            context = model.context(cls, tokens, batch, present, list(sources))
        # CMR reads the others without the CLS of every instrument
        context = Context(*(one[:, 1:] for one in context))
        cmr, drawn = rolled_attention(
            model.decoders[query],
            context,
            batch[query].position,
            batch[query].acquisition,
            present[query],
        )
        drawn = drawn.cpu().numpy()
        readable = int(context.readable.sum())
        attended = {query: sample[query] | {"cmr": cmr[0].cpu().numpy()}}
        start = 0
        for name in sources:
            # Each instrument's CLS, then its tokens
            taken = drawn[start : start + 1 + batch[name].values.shape[1]]
            start += taken.size
            attended[name] = sample[name] | {
                "attention": taken[1:] * readable,
                "share": taken.sum(),
                "cls": taken[0],
            }
        return attended

    attention = {
        query: cmr_attention(query, [name for name in sizes if name != query])
        for query in sizes
    }
    ranged = {
        query: cmr_attention(query, [other])
        for query, other in zip(ranging, ranging[::-1], strict=True)
    }
    return reconstruction, attention, ranged, tokens
