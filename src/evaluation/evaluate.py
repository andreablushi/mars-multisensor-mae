"""Embedding the tiles a trained model never read, and logging what it comes to."""

from __future__ import annotations

import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader
from wandb.sdk.wandb_run import Run

from architecture.mae import CrossSensorMAE
from architecture.tokens import CLS, GAP, Tokens, global_average_pooling
from configs.paths import (
    ATTENTION_FILE,
    RANGING_FILE,
    RECONSTRUCTION_FILE,
    TOKENS_FILE,
    results_path,
)
from dataset.loader import tile_loader
from dataset.store import DatasetBuild
from evaluation.reconstruct import reconstruction_and_attention
from evaluation.results import write_tile_arrays, write_tile_vectors
from logs.evaluation import log_evaluation

EVERY_INSTRUMENT = "all"
EVALUATION_BATCH_SIZE = 1


def evaluate(
    model: CrossSensorMAE,
    build: DatasetBuild,
    sizes: Mapping[str, Mapping[str, int]],
    shapes: Mapping[str, tuple[int, ...]],
    tiles: Mapping[str, str],
    mask_ratio: float,
    ranging: Sequence[str],
    seed: int,
    workers: int,
    device: torch.device,
    run: Run,
    run_name: str,
) -> None:
    """Write and log every tile's vectors, the scores and the drawn tiles.

    Args:
        model: The model, on the device.
        build: The build the labelled tiles are read from.
        sizes: How far a patch of each sensor runs along each axis it is cut on.
        shapes: The shape of one patch of each instrument as the model reads it.
        tiles: The tiles drawn as reconstructions and attention maps, by what they show.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        ranging: The two ranging instruments, each also rebuilt from the other alone.
        seed: What fixes the masks.
        workers: How many processes read tiles beside the model.
        device: Where the model runs.
        run: The tracked run.
        run_name: What the run is called, which names its results and artifact.
    """
    by_tile = build.read_observation_metadata()
    loader = tile_loader(
        build, by_tile, sizes, None, 1.0, seed, EVALUATION_BATCH_SIZE, workers, False
    )
    vectors = tile_vectors(model, loader, device)
    classes = build.read_label_by_tile()
    directory = results_path(run_name).parent
    shutil.rmtree(directory, ignore_errors=True)
    write_tile_vectors(results_path(run_name), classes, vectors)
    reconstructions, attentions, rangings, tokens = {}, {}, {}, {}
    for place, tile in tiles.items():
        reconstructions[place], by_query, ranged, tokens[place] = (
            reconstruction_and_attention(
                model,
                build,
                by_tile[tile],
                sizes,
                shapes,
                mask_ratio,
                ranging,
                seed,
                device,
            )
        )
        attentions |= {(place, query): one for query, one in by_query.items()}
        rangings |= {(place, query): one for query, one in ranged.items()}
    arrays = {
        **{
            RECONSTRUCTION_FILE.format(key): one for key, one in reconstructions.items()
        },
        **{TOKENS_FILE.format(key): one for key, one in tokens.items()},
        **{ATTENTION_FILE.format(*key): one for key, one in attentions.items()},
        **{RANGING_FILE.format(*key): one for key, one in rangings.items()},
    }
    for name, held in arrays.items():
        write_tile_arrays(results_path(run_name, name), held)
    log_evaluation(
        run,
        run_name,
        vectors,
        classes,
        reconstructions,
        attentions,
        rangings,
        tokens,
        directory,
    )


def tile_vectors(
    model: CrossSensorMAE, loader: DataLoader, device: torch.device
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Return one vector per tile, pooling and view: each instrument's, and all.

    Args:
        model: The model, on the device.
        loader: The tiles to read, in batches, each tile read whole.
        device: Where the model runs.

    Returns:
        vectors: Each tile's unit vector, keyed by CLS or GAP, then by instrument or
            EVERY_INSTRUMENT, then by tile, CLS of every instrument being the
            model's own and GAP's the mean of the instruments'. (D,)
    """
    model.eval()
    vectors = defaultdict(lambda: defaultdict(dict))
    with torch.no_grad():
        for batch, identities in loader:
            batch = {
                name: Tokens(*(one.to(device, non_blocking=True) for one in tokens))
                for name, tokens in batch.items()
            }
            present = {name: tokens.present for name, tokens in batch.items()}
            cls, tokens = model.shared_tokens(batch, present)  # (B, 1 + K, D) each
            pooled = {
                CLS: {
                    name: functional.normalize(held[:, 0], dim=-1)
                    for name, held in tokens.items()
                },
                GAP: {
                    name: global_average_pooling(held[:, 1:], present[name])
                    for name, held in tokens.items()
                },
            }
            pooled[CLS][EVERY_INSTRUMENT] = functional.normalize(cls, dim=-1)
            stacked = torch.stack(list(pooled[GAP].values()), dim=1)  # (B, M, D)
            pooled[GAP][EVERY_INSTRUMENT] = global_average_pooling(
                stacked, torch.ones(stacked.shape[:2], dtype=torch.bool, device=device)
            )
            for pooling, by_view in pooled.items():
                for view, held in by_view.items():
                    vectors[pooling][view].update(
                        zip(identities, held.cpu().numpy(), strict=True)
                    )
    return {pooling: dict(by_view) for pooling, by_view in vectors.items()}
