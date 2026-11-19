"""Embedding the tiles a trained model never read, and scoring what it comes to."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader
from wandb.sdk.wandb_run import Run

from architecture.mae import CrossSensorMAE, global_tokens
from architecture.models.context import Pooling
from dataset.loader import tile_loader
from dataset.models.patch import PatchLayout
from dataset.store import DatasetBuild
from evaluation.metrics import metrics_by_view, modality_gap_by_pooling
from evaluation.models.vectors import TileVectors
from evaluation.results import write_vectors
from logs.evaluation import log_evaluation


def evaluate(
    model: CrossSensorMAE,
    build: DatasetBuild,
    layout: PatchLayout,
    batch_size: int,
    neighbours: int,
    rank_build: DatasetBuild,
    rank_tiles: Sequence[str],
    budget: Mapping[str, int],
    rank_batch_size: int,
    seed: int,
    workers: int,
    device: torch.device,
    run: Run,
    run_name: str,
) -> None:
    """Write every tile's vectors and log the scores they come to.

    Args:
        model: The model, on the device.
        build: The build the labelled tiles are read from.
        layout: How each instrument is cut into patches, and the shape the model
            reads one in.
        batch_size: How many tiles one batch of the vector pass holds.
        neighbours: How many neighbours the kNN vote reads.
        rank_build: The training build RankMe's tiles are read from.
        rank_tiles: The training tiles RankMe reads, as the source dataset in the
            paper.
        budget: How many patches of each instrument a RankMe tile keeps, around a
            fixed centre.
        rank_batch_size: How many RankMe tiles one batch holds.
        seed: What fixes each RankMe tile's centre.
        workers: How many processes read tiles beside the model.
        device: Where the model runs.
        run: The tracked run.
        run_name: What the run is called, which names its results and artifact.
    """
    by_tile = build.read_observation_metadata()
    loader = tile_loader(
        build, by_tile, layout, None, 1.0, seed, batch_size, workers, False
    )
    vectors = [
        pooled.sorted_by_identity()
        for pooled in tile_vectors(model, loader, device, False)
    ]
    # RankMe reads training samples, cut as training cuts them and not kept
    raw = tile_vectors(
        model,
        tile_loader(
            rank_build,
            rank_tiles,
            layout,
            budget,
            1.0,
            seed,
            rank_batch_size,
            workers,
            False,
            cache=False,
        ),
        device,
        True,
    )
    classes = build.read_label_by_tile()
    log_evaluation(
        run,
        run_name,
        metrics_by_view(vectors, raw, classes, neighbours),
        modality_gap_by_pooling(vectors),
        vectors,
        write_vectors(run_name, classes, vectors),
    )


def tile_vectors(
    model: CrossSensorMAE, loader: DataLoader, device: torch.device, raw: bool
) -> list[TileVectors]:
    """Return one vector per tile, pooling and view, unit or raw: each instrument, all.

    Args:
        model: The model, on the device.
        loader: The tiles to read, in batches, each tile read whole.
        device: Where the model runs.
        raw: Whether the vectors are kept before any normalisation, as RankMe reads.

    Returns:
        vectors: CLS, then GAP, rows in the order read; fused is the model's CLS, and
            for GAP the mean of the instruments'.
    """
    tiles = []
    rows = defaultdict(lambda: defaultdict(list))
    fused = defaultdict(list)
    with torch.no_grad():
        for batch, identities in loader:
            tiles.extend(identities)
            batch = {name: tokens.to(device) for name, tokens in batch.items()}
            present = {name: tokens.present for name, tokens in batch.items()}
            cls, contexts = model.embed_tokens(batch, present)
            if raw:
                pooled = {
                    Pooling.CLS: {
                        name: held.tokens[:, 0] for name, held in contexts.items()
                    },
                    Pooling.GAP: {
                        name: (held.tokens[:, 1:] * present[name].unsqueeze(-1)).sum(1)
                        / present[name].sum(1, keepdim=True).clamp(min=1)
                        for name, held in contexts.items()
                    },
                }
                overall = {
                    Pooling.CLS: cls,
                    Pooling.GAP: torch.stack(list(pooled[Pooling.GAP].values())).mean(
                        0
                    ),
                }
            else:
                pooled = {
                    pooling: global_tokens(contexts, present, pooling)
                    for pooling in (Pooling.CLS, Pooling.GAP)
                }
                overall = {
                    Pooling.CLS: functional.normalize(cls, dim=-1),
                    Pooling.GAP: functional.normalize(
                        torch.stack(list(pooled[Pooling.GAP].values()), dim=1).sum(1),
                        dim=-1,
                    ),
                }
            for pooling, by_name in pooled.items():
                for name, held in by_name.items():
                    rows[pooling][name].append(held.float().cpu().numpy())
                fused[pooling].append(overall[pooling].float().cpu().numpy())
    return [
        TileVectors(
            pooling,
            tiles,
            {name: np.concatenate(held) for name, held in by_name.items()},
            np.concatenate(fused[pooling]),
        )
        for pooling, by_name in rows.items()
    ]
