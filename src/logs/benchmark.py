"""Logging what one benchmark comes to: each head's steps, its scores, drawn tiles."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import wandb
from wandb.sdk.wandb_run import Run

from benchmarking.evaluation.metrics import COMBOS
from logs.tracker import artifact_name

BENCHMARK = "benchmark"

STEP = f"{BENCHMARK}/step"


def log_step(run: Run, prefix: str, step: int, loss: float) -> None:
    """Log one training step of one head, its loss against the step.

    Args:
        run: The tracked run.
        prefix: Where the head's steps are logged.
        step: Which step it is, from 1.
        loss: The step's training loss.
    """
    run.log({STEP: step, f"{prefix}/train/loss": loss})


def log_epoch(
    run: Run,
    prefix: str,
    step: int,
    scores: Mapping[str, Mapping[str, float]],
) -> None:
    """Log one epoch of one head, its validation scores against its last step.

    Args:
        run: The tracked run.
        prefix: Where the head's steps are logged.
        step: The epoch's last step.
        scores: Each validation metric, keyed by depth, then by metric.
    """
    run.define_metric(f"{prefix}/*", step_metric=STEP)
    run.log(
        {STEP: step}
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
    confusion: Mapping[str, np.ndarray],
    images: Mapping[str, tuple[np.ndarray, str]],
    path: Path,
) -> None:
    """Log the test scores and confusion, the drawn tiles, and the predictions.

    Args:
        run: The tracked run.
        run_name: What the tracked run is called, which names its artifact.
        scores: Each test metric, keyed by protocol and task, then by depth and
            metric.
        confusion: How many test tiles of each SWIM combo each classifier
            predicts as each combo, keyed by protocol. (len(COMBOS), len(COMBOS))
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
            f"{BENCHMARK}/confusion": wandb.Table(
                columns=["protocol", "truth", "predicted", "count"],
                data=[
                    [protocol, truth, predicted, int(counts[row, column])]
                    for protocol, counts in confusion.items()
                    for row, truth in enumerate(COMBOS)
                    for column, predicted in enumerate(COMBOS)
                ],
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
