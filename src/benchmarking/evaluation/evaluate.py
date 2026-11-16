"""Predicting the test tiles with every protocol's model, and logging the scores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from wandb.sdk.wandb_run import Run

from benchmarking.architecture.model import PROTOCOLS, BenchmarkModel
from benchmarking.evaluation.metrics import rmse_and_r2
from dataset.loader import tile_loader
from dataset.store import DatasetBuild
from evaluation.reconstruct import drawn_arrays
from logs.benchmark import log_benchmark
from logs.images.ice import ice_image
from training.step import autocast_dtype


def evaluate(
    models: Mapping[str, Mapping[str, BenchmarkModel]],
    build: DatasetBuild,
    test: Sequence[str],
    targets: Mapping[str, np.ndarray],
    instruments: Sequence[str],
    sizes: Mapping[str, Mapping[str, int]],
    shapes: Mapping[str, tuple[int, ...]],
    budget: Mapping[str, int],
    reach_m: float,
    batch_size: int,
    tiles: Sequence[str],
    shown: str,
    seed: int,
    workers: int,
    device: torch.device,
    run: Run,
) -> None:
    """Log every model's test scores and the drawn tiles.

    Args:
        models: Each protocol's model, on the device, trained, keyed by the share
            of the tiles it was trained on, the largest last.
        build: The build the test tiles are read from.
        test: The test tiles.
        targets: SWIM's value of every point for each target, keyed by the
            tile over at least the test tiles. (P * P, Z)
        instruments: The instruments the models read.
        sizes: How far a patch of each sensor runs along each axis it is cut on.
        shapes: The shape of one patch of each instrument as the model reads it.
        budget: How many patches of each instrument a test crop keeps.
        reach_m: How near a read patch of a crop a point is predicted, in metres.
        batch_size: How many tiles one batch holds, each read as its covering crops.
        tiles: The test tiles drawn as SWIM's grid beside the models', in
            qualitative order.
        shown: The instrument whose every observation of a tile is drawn beneath.
        seed: What fixes each fixed cut.
        workers: How many processes read tiles beside the models.
        device: Where the models run.
        run: The tracked run.
    """
    read = {name: sizes[name] for name in instruments}
    loader = tile_loader(
        build,
        test,
        read,
        shapes,
        budget,
        1.0,
        seed,
        batch_size,
        workers,
        False,
        reach_m=reach_m,
    )
    predictions = {
        share: {protocol: {} for protocol in held} for share, held in models.items()
    }
    still = next(protocol for protocol, tuned in PROTOCOLS.items() if not tuned)
    with torch.no_grad():
        for batch, identities, scored in loader:
            # Each point is the mean of the crops of its tile that predict it
            weights = torch.as_tensor(scored, dtype=torch.float32).unsqueeze(-1)
            crops = {
                tile: [at for at, one in enumerate(identities) if one == tile]
                for tile in dict.fromkeys(identities)
            }
            batch = {name: tokens.to(device) for name, tokens in batch.items()}
            with torch.autocast(device.type, dtype=autocast_dtype(device)):
                # The frozen MAE is the same under every share, so it reads once
                frozen = next(iter(models.values()))[still].tokens(batch)
                for share, held in models.items():
                    for protocol, model in held.items():
                        read_tokens = (
                            model.tokens(batch) if PROTOCOLS[protocol] else frozen
                        )
                        predicted = model.probe(read_tokens).float().cpu() * weights
                        predictions[share][protocol].update(
                            {
                                tile: (
                                    predicted[at].sum(0) / weights[at].sum(0)
                                ).numpy()
                                for tile, at in crops.items()
                            }
                        )
    drawn = predictions[list(models)[-1]]
    by_tile = build.read_observation_metadata()
    images = [
        ice_image(
            drawn_arrays(
                build, shown, by_tile[tile][shown], sizes[shown], shapes[shown]
            ),
            targets[tile],
            {protocol: held[tile] for protocol, held in drawn.items()},
        )
        for tile in tiles
    ]
    scores = {share: rmse_and_r2(held, targets) for share, held in predictions.items()}
    log_benchmark(run, scores, images)
