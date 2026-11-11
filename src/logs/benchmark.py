"""Logging what one benchmark comes to: its steps, its scores, its drawn tiles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import wandb
from wandb.sdk.wandb_run import Run

BENCHMARK = "benchmark"

FINETUNE = f"{BENCHMARK}-finetune"

HEADS = "heads"

STEP = f"{BENCHMARK}/step"

RESULTS_STEP = 10**6


def score_rows(
    scores: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> list[tuple[str, str, str, float]]:
    """Return every score as one row.

    Args:
        scores: Each metric, keyed by protocol, then by target and metric.

    Returns:
        rows: Each score's protocol, target, metric and value.
    """
    return [
        (protocol, target, metric, value)
        for protocol, by_target in scores.items()
        for target, held in by_target.items()
        for metric, value in held.items()
    ]


def log_step(run: Run, step: int, share: str, losses: Mapping[str, float]) -> None:
    """Log one training step, every protocol's loss against the step.

    Args:
        run: The tracked run.
        step: Which step it is, from 1.
        share: What the share of the tiles trained on is logged as.
        losses: The training loss, keyed by protocol.
    """
    run.log(
        {STEP: step}
        | {
            f"{BENCHMARK}/{protocol}/{share}/train/loss": one
            for protocol, one in losses.items()
        }
    )


def log_epoch(
    run: Run,
    step: int,
    share: str,
    scores: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> None:
    """Log one validation, every protocol's scores against the step.

    Args:
        run: The tracked run.
        step: The step validated at.
        share: What the share of the tiles trained on is logged as.
        scores: Each validation metric, keyed by protocol, then by target and metric.
    """
    run.define_metric(f"{BENCHMARK}/*", step_metric=STEP)
    run.log(
        {STEP: step}
        | {
            f"{BENCHMARK}/{protocol}/{share}/validation/{target}/{metric}": value
            for protocol, target, metric, value in score_rows(scores)
        }
    )


def log_benchmark(
    run: Run,
    scores: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]],
    images: Sequence[tuple[np.ndarray, str]],
) -> None:
    """Log the test scores and drawn tiles at one step for every run.

    Args:
        run: The tracked run.
        scores: Each test metric, keyed by the share of the tiles trained on, then
            by protocol, target and metric.
        images: Each drawn tile's image and caption, logged as qualitative 1 on.
    """
    rows = [(share, *row) for share, held in scores.items() for row in score_rows(held)]
    run.summary.update(
        {
            f"{BENCHMARK}/{protocol}/{share}/test/{target}/{metric}": value
            for share, protocol, target, metric, value in rows
        }
    )
    run.log(
        {
            f"{BENCHMARK}/scores": wandb.Table(
                columns=["share", "protocol", "target", "metric", "value"],
                data=[list(row) for row in rows],
            ),
            **{
                f"{BENCHMARK}/qualitative-{at}": wandb.Image(image, caption=caption)
                for at, (image, caption) in enumerate(images, 1)
            },
        },
        step=RESULTS_STEP,
    )
