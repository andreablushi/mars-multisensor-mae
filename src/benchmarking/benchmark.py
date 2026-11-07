"""Probing and fine-tuning heads on SWIM's ice, and logging what they come to."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from analysis.ground_truth.swim.models import IceLabel
from wandb.sdk.wandb_run import Run

from architecture.mae import CrossSensorMAE
from benchmarking.architecture.classifier import Classifier
from benchmarking.architecture.lora import adapted_model
from benchmarking.architecture.segmenter import Segmenter
from benchmarking.evaluation.metrics import DEPTHS, ice_scores
from benchmarking.evaluation.results import write_predictions
from benchmarking.finetune.finetune import finetune
from dataset.loader import ice_split, tile_loader
from dataset.store import DatasetBuild
from evaluation.reconstruct import drawn_arrays
from logs.benchmark import BENCHMARK, log_benchmark
from logs.images.ice import ice_image

PROTOCOLS = {"linear-probe": False, "fine-tune": True}

TITLES = {"linear-probe": "Linear Probe", "fine-tune": "Fine-tuning"}


def benchmark(
    model: CrossSensorMAE,
    build: DatasetBuild,
    labels: Mapping[str, IceLabel],
    instruments: Sequence[str],
    sizes: Mapping[str, Mapping[str, int]],
    shapes: Mapping[str, tuple[int, ...]],
    dim: int,
    heads: int,
    segmenter_depth: int,
    folds: int,
    block_deg: float,
    pooling: str,
    budget: Mapping[str, int],
    budget_scale: float,
    batch_size: int,
    whole_batch_size: int,
    epochs: int,
    warmup_epochs: int,
    head_learning_rate: float,
    lora_rank: int,
    lora_alpha: float,
    lora_learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    tiles: Mapping[str, str],
    shown: str,
    seed: int,
    workers: int,
    device: torch.device,
    run: Run,
    run_name: str,
) -> None:
    """Probe and fine-tune both heads, then log their test scores and the drawn tiles.

    Args:
        model: The model, on the device, left as it is.
        build: The build the labelled tiles are read from.
        labels: Each tile's ice at both depths, keyed by the tile, over any build.
        instruments: The instruments the heads read.
        sizes: How far a patch of each sensor runs along each axis it is cut on.
        shapes: The shape of one patch of each instrument as the model reads it.
        dim: The token width the model hands its tokens at.
        heads: How many attention heads the map head runs.
        segmenter_depth: How many cross-attention blocks the map head stacks.
        folds: How many equal folds the labelled tiles are dealt into, the first
            testing, the second validating and the rest training.
        block_deg: How wide a square of the map is, in degrees, each one dealt whole.
        pooling: What the tile head reads of a tile, "GAP" or "CLS".
        budget: How many patches of each instrument a training tile keeps.
        budget_scale: The most a training batch's budget is multiplied by.
        batch_size: How many tiles one training batch holds.
        whole_batch_size: How many tiles one batch read whole holds.
        epochs: How many epochs the heads train for, each kept at its best
            validation RMSE.
        warmup_epochs: How many epochs the rate climbs before the cosine decay.
        head_learning_rate: The head's peak learning rate.
        lora_rank: The rank of each LoRA update of the encoders when fine-tuned.
        lora_alpha: What each LoRA update is scaled by before it is divided by its
            rank.
        lora_learning_rate: The LoRA updates' peak learning rate.
        weight_decay: The AdamW weight decay.
        max_grad_norm: The norm every step's gradient is clipped to.
        tiles: The test tiles drawn as SWIM's grid beside the map heads', by what
            they show.
        shown: The instrument whose every observation of a tile is drawn beneath.
        seed: What fixes the split, the order and each fixed cut.
        workers: How many processes read tiles beside the model.
        device: Where the model runs.
        run: The tracked run.
        run_name: What the tracked run is called, which names its predictions.
    """
    by_tile = build.read_observation_metadata()
    labels = {tile: one for tile, one in labels.items() if tile in by_tile}
    training, validation, test = ice_split(labels, folds, block_deg, seed)
    read = {name: sizes[name] for name in instruments}
    grids = {
        tile: np.stack(
            [getattr(one, f"consistency_{depth}") for depth in DEPTHS], -1
        ).astype(np.float32)
        for tile, one in labels.items()
    }
    targets = {
        "tile": {tile: np.median(grid, 0) for tile, grid in grids.items()},
        "map": grids,
    }
    tested = tile_loader(
        build, test, read, shapes, None, 1.0, seed, whole_batch_size, workers, False
    )
    model.requires_grad_(False)
    consistency = {}
    for protocol, tuned in PROTOCOLS.items():
        adapted = adapted_model(model, lora_rank, lora_alpha) if tuned else model
        predicted = finetune(
            adapted,
            {
                "tile": Classifier(dim, len(DEPTHS), pooling, not tuned).to(device),
                "map": Segmenter(dim, heads, segmenter_depth, len(DEPTHS)).to(device),
            },
            build,
            training,
            validation,
            tested,
            targets,
            read,
            shapes,
            budget,
            budget_scale,
            batch_size,
            whole_batch_size,
            workers,
            epochs,
            warmup_epochs,
            head_learning_rate,
            lora_learning_rate,
            weight_decay,
            max_grad_norm,
            seed,
            device,
            run,
            f"{BENCHMARK}/{protocol}",
        )
        consistency |= {(protocol, task): held for task, held in predicted.items()}
    images = {
        place: ice_image(
            drawn_arrays(
                build, shown, by_tile[tile][shown], sizes[shown], shapes[shown]
            ),
            {"Ground Truth": grids[tile]}
            | {
                TITLES[protocol]: consistency[protocol, "map"][tile]
                for protocol in PROTOCOLS
            },
        )
        for place, tile in tiles.items()
    }
    log_benchmark(
        run,
        run_name,
        {
            (protocol, task): ice_scores(held, targets[task], task == "map")
            for (protocol, task), held in consistency.items()
        },
        images,
        write_predictions(run_name, consistency),
    )
