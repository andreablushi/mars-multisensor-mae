"""Logging what one benchmark comes to: each head's epochs, its scores, drawn tiles."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import wandb
from wandb.sdk.wandb_run import Run

from logs.tracker import artifact_name

BENCHMARK = "benchmark"

EPOCH = f"{BENCHMARK}/epoch"


def log_epoch(
    run: Run,
    prefix: str,
    epoch: int,
    loss: float,
    scores: Mapping[str, Mapping[str, float]],
) -> None:
    """Log one epoch of one head, against the epoch: its loss and validation scores.

    Args:
        run: The tracked run.
        prefix: Where the head's epochs are logged.
        epoch: Which epoch it is, from 1.
        loss: The epoch's mean training loss.
        scores: Each validation metric, keyed by depth, then by metric.
    """
    run.define_metric(f"{prefix}/*", step_metric=EPOCH)
    run.log(
        {EPOCH: epoch, f"{prefix}/train/loss": loss}
        | {
            f"{prefix}/validation/{depth}/{metric}": value
            for depth, held in scores.items()
            for metric, value in held.items()
        }
    )


def log_benchmark(
    run: Run,
    run_name: str,
    scores: Mapping[tuple[str, str], Mapping[str, Mapping[str, float]]],
    images: Mapping[str, tuple[np.ndarray, str]],
    path: Path,
) -> None:
    """Log every test score, the drawn tiles, and the predictions as an artifact.

    Args:
        run: The tracked run.
        run_name: What the tracked run is called, which names its artifact.
        scores: Each test metric, keyed by protocol and task, then by depth and
            metric.
        images: Each drawn tile's image and caption, keyed by what it shows.
        path: Where the predictions were written.
    """
    rows = [
        [protocol, task, depth, metric, value]
        for (protocol, task), by_depth in scores.items()
        for depth, held in by_depth.items()
        for metric, value in held.items()
    ]
    run.summary.update(
        {
            f"{BENCHMARK}/{protocol}/{task}/test/{depth}/{metric}": value
            for protocol, task, depth, metric, value in rows
        }
    )
    run.log(
        {
            f"{BENCHMARK}/scores": wandb.Table(
                columns=["protocol", "task", "depth", "metric", "value"], data=rows
            ),
            **{
                f"{BENCHMARK}/grids/{place}": wandb.Image(image, caption=caption)
                for place, (image, caption) in images.items()
            },
        }
    )
    artifact = wandb.Artifact(artifact_name(BENCHMARK, run_name), type=BENCHMARK)
    artifact.add_file(str(path))
    run.log_artifact(artifact)
