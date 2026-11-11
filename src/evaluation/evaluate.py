"""Embedding the tiles a trained model never read, and logging what it comes to."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader
from wandb.sdk.wandb_run import Run

from architecture.mae import CrossSensorMAE, pooled_tokens
from configs.schema import DrawnTile
from dataset.loader import tile_loader
from dataset.store import DatasetBuild
from evaluation.metrics import metrics_by_view
from evaluation.reconstruct import reconstruction_and_attention
from evaluation.results import write_results
from logs.evaluation import log_evaluation


def evaluate(
    model: CrossSensorMAE,
    build: DatasetBuild,
    sizes: Mapping[str, Mapping[str, int]],
    shapes: Mapping[str, tuple[int, ...]],
    batch_size: int,
    neighbours: int,
    tiles: Mapping[str, DrawnTile],
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
        batch_size: How many tiles one batch of the vector pass holds.
        neighbours: How many neighbours the kNN vote reads.
        tiles: The tiles drawn as reconstructions and attention maps, by what they
            show, with each instrument's observation shown.
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
        build, by_tile, sizes, shapes, None, 1.0, seed, batch_size, workers, False
    )
    vectors = tile_vectors(model, loader, device)
    classes = build.read_label_by_tile()
    reconstructions, rangings = {}, {}
    for place, drawn in tiles.items():
        by_query, ranged = reconstruction_and_attention(
            model,
            build,
            by_tile[drawn.tile],
            drawn.shown,
            sizes,
            shapes,
            mask_ratio,
            ranging,
            seed,
            device,
        )
        reconstructions |= {(place, query): one for query, one in by_query.items()}
        rangings |= {(place, query): one for query, one in ranged.items()}
    directory = write_results(run_name, classes, vectors, reconstructions, rangings)
    log_evaluation(
        run,
        run_name,
        metrics_by_view(vectors, classes, neighbours),
        reconstructions,
        rangings,
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
            "all", then by tile, CLS of every instrument being the
            model's own and GAP's the mean of the instruments'. (D,)
    """
    vectors = defaultdict(lambda: defaultdict(dict))
    with torch.no_grad():
        for batch, identities in loader:
            batch = {name: tokens.to(device) for name, tokens in batch.items()}
            present = {name: tokens.present for name, tokens in batch.items()}
            cls, contexts = model.embed_tokens(batch, present)
            pooled = {
                pooling: pooled_tokens(contexts, present, pooling)
                for pooling in ("CLS", "GAP")
            }
            pooled["CLS"]["all"] = functional.normalize(cls, dim=-1)
            pooled["GAP"]["all"] = functional.normalize(
                torch.stack(list(pooled["GAP"].values()), dim=1).sum(1), dim=-1
            )
            for pooling, by_view in pooled.items():
                for view, held in by_view.items():
                    vectors[pooling][view].update(
                        zip(identities, held.cpu().numpy(), strict=True)
                    )
    return {pooling: dict(by_view) for pooling, by_view in vectors.items()}
