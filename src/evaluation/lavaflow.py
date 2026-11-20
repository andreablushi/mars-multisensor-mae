"""Retrieving the SHARAD crops that show a buried lava flow, by their vectors alone."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import torch
from lavaflow.models import REFLECTOR
from mars_multisensor_dataset.building.configs import sharad
from torch.utils.data import DataLoader
from wandb.sdk.wandb_run import Run

from architecture.mae import CrossSensorMAE, global_tokens
from architecture.models.context import Pooling
from configs.load import named_sample
from dataset.loader import tile_loader
from dataset.models.patch import PatchLayout
from dataset.patches import drawn_arrays
from dataset.store import DatasetBuild
from evaluation.metrics import ranked_neighbours, retrieval_metrics
from evaluation.models.metrics import Score
from evaluation.models.vectors import TileVectors
from logs.evaluation import log_lavaflow
from logs.images.lavaflow import lavaflow_image


def evaluate_lavaflow(
    model: CrossSensorMAE,
    build: DatasetBuild,
    size: Mapping[str, int],
    shape: tuple[int, ...],
    batch_size: int,
    queries: int,
    top_k: int,
    pooling: Pooling,
    seed: int,
    workers: int,
    device: torch.device,
    run: Run,
) -> None:
    """Log how early each reflector crop retrieves the others, and draw a few of them.

    Args:
        model: The model, on the device.
        build: The lava flow build, SHARAD crops and their verdicts.
        size: How far a SHARAD patch runs along each axis it is cut on.
        shape: The shape of one SHARAD patch as the model reads it.
        batch_size: How many crops one batch of the vector pass holds.
        queries: How many reflector crops are drawn beside what they retrieve.
        top_k: How many crops each drawn query shows it retrieved.
        pooling: What the drawn queries are ranked by, "CLS", "GAP" or "GAP-surface".
        seed: What fixes the drawn queries.
        workers: How many processes read crops beside the model.
        device: Where the model runs.
        run: The tracked run.
    """
    name = sharad.LAYOUT.instrument
    crops = build.read_lavaflow_crops()
    by_tile = build.read_observation_metadata()
    rows = {
        one.crop: [
            row for row in by_tile[one.tile][name] if row.identifier == one.track
        ]
        for one in crops
    }
    by_crop = replace(
        build, records={crop: {name: held} for crop, held in rows.items()}
    )
    vectors = crop_vectors(
        model,
        tile_loader(
            by_crop,
            rows,
            PatchLayout({name: size}, {name: shape}),
            None,
            1.0,
            seed,
            batch_size,
            workers,
            False,
        ),
        device,
    )
    verdicts = np.array([one.verdict for one in crops])
    tracks, tiles = (
        np.array([getattr(one, key) for one in crops]) for key in ("track", "tile")
    )
    same_track = tracks[:, None] == tracks  # (N, N)
    galleries = {
        "track-out": same_track,
        "tile-out": same_track | (tiles[:, None] == tiles),
    }
    queried = verdicts == REFLECTOR  # (N,)
    similarities = {}
    for pooled in vectors:
        row = {crop: at for at, crop in enumerate(pooled.identities)}
        stacked = pooled.instruments[name][[row[one.crop] for one in crops]].astype(
            np.float64
        )
        similarities[pooled.pooling] = stacked @ stacked.T  # (N, N)
    neighbours = {
        pooled: ranked_neighbours(held) for pooled, held in similarities.items()
    }
    scores = [
        Score(
            pooled,
            gallery,
            "mAP",
            float(retrieval_metrics(held, verdicts, hidden)["mAP"][queried].mean()),
        )
        for pooled, held in neighbours.items()
        for gallery, hidden in galleries.items()
    ]
    drawn = named_sample(
        np.flatnonzero(queried).tolist(), queries, seed, "lavaflow/queries"
    )
    ranked = np.where(same_track, -np.inf, similarities[pooling])  # (N, N)
    images = []
    for query in drawn:
        retrieved = np.argsort(-ranked[query], kind="stable")[:top_k]
        shown = [query, *retrieved]
        images.append(
            lavaflow_image(
                [
                    drawn_arrays(build, name, rows[crops[at].crop], size, shape)
                    for at in shown
                ],
                [crops[at].crop for at in shown],
                verdicts[retrieved].tolist(),
                ranked[query, retrieved].tolist(),
            )
        )
    log_lavaflow(run, scores, images)


def crop_vectors(
    model: CrossSensorMAE, loader: DataLoader, device: torch.device
) -> list[TileVectors]:
    """Return one unit vector per SHARAD crop and pooling, its ground centred on itself.

    Args:
        model: The model, on the device.
        loader: The crops to read, in batches, each read whole, SHARAD alone.
        device: Where the model runs.

    Returns:
        vectors: CLS, GAP and GAP-surface, the mean of each crop's top row of
            patches, their one view SHARAD, rows in the order read.
    """
    crops = []
    rows = defaultdict(list)
    with torch.no_grad():
        for batch, identities in loader:
            crops.extend(identities)
            ((name, tokens),) = batch.items()
            tokens = tokens.to(device)
            present = tokens.present  # (B, K)
            counted = present.unsqueeze(-1).float()  # (B, K, 1)
            ground = tokens.position[..., :2]  # (B, K, 2)
            centre = (ground * counted).sum(1, keepdim=True) / counted.sum(
                1, keepdim=True
            )
            tokens = tokens._replace(
                position=torch.cat([ground - centre, tokens.position[..., 2:]], dim=-1)
            )
            _, contexts = model.embed_tokens({name: tokens}, {name: present})
            delay = tokens.position[..., 2].masked_fill(~present, torch.inf)  # (B, K)
            surface = present & (delay == delay.min(dim=1, keepdim=True).values)
            pooled = {
                Pooling.CLS: global_tokens(contexts, {name: present}, Pooling.CLS),
                Pooling.GAP: global_tokens(contexts, {name: present}, Pooling.GAP),
                Pooling.GAP_SURFACE: global_tokens(
                    contexts, {name: surface}, Pooling.GAP
                ),
            }
            for pooling, held in pooled.items():
                rows[pooling].append(held[name].cpu().numpy())
    return [
        TileVectors(pooling, crops, {name: np.concatenate(held)}, None)
        for pooling, held in rows.items()
    ]
