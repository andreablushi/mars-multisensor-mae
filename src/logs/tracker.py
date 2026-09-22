"""Tracking a run on Weights & Biases, every metric named in one place."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict

import numpy as np
import plotly.graph_objects as go
import wandb
from dotenv import load_dotenv
from torch import Tensor
from wandb.sdk.wandb_run import Run

from config.paths import REPO_ROOT
from config.schema import Config


def start_logging(config: Config, stage: str, facts: Mapping[str, object]) -> Run:
    """Return the tracked run every metric of one stage is logged to.

    Args:
        config: What the run reads, trains and how, which the run is recorded with.
        stage: What the stage is called, which tells a training from an evaluation.
        facts: What only the assembled run knows, recorded beside the config.

    Returns:
        run: The run, named and grouped by the config's run name so the training and
            the evaluation of one model sit together, its place read from the .env.
    """
    load_dotenv(REPO_ROOT / ".env")
    run = wandb.init(
        config=asdict(config),
        name=config.run_name,
        group=config.run_name,
        job_type=stage,
    )
    run.config.update(dict(facts))
    return run


def log_step(run: Run, step: int, terms: Mapping[str, Tensor]) -> None:
    """Log one training step.

    Args:
        run: The tracked run.
        step: Which step of the whole run it is.
        terms: Every loss term, keyed as the loss names them.
    """
    run.log(
        {f"train_{name}": value.item() for name, value in terms.items()},
        step=step,
    )


def log_validation(run: Run, step: int, metrics: Mapping[str, float]) -> None:
    """Log one validation.

    Args:
        run: The tracked run.
        step: Which step of the whole run it was measured after.
        metrics: Every loss term over the validation split, keyed as they name them.
    """
    run.log({f"validation_{name}": value for name, value in metrics.items()}, step=step)


def log_summary(run: Run, summary: Mapping[str, object]) -> None:
    """Write what the whole run came to.

    Args:
        run: The tracked run.
        summary: What it came to, keyed as it is recorded.
    """
    run.summary.update(dict(summary))


def log_latent_space(
    run: Run,
    metrics: Mapping[str, float],
    classes: Sequence[str],
    distances: Sequence[Sequence[float]],
    tiles: Sequence[str],
    labels: Sequence[str],
    projection: np.ndarray,
) -> None:
    """Log what a model's latent space came to over the labelled tiles.

    Args:
        run: The tracked run.
        metrics: Every number measured, keyed as it is logged.
        classes: The classes, in the order the distances hold them.
        distances: The mean distance between the tiles of two classes. (C, C)
        tiles: The tiles, in the order the projection holds them.
        labels: The class each tile carries, in the same order.
        projection: Where UMAP lays each tile on a plane. (T, 2)
    """
    rows = [[name, *map(float, row)] for name, row in zip(classes, distances)]
    table = wandb.Table(columns=["class", *classes], data=rows)
    held = np.asarray(labels)
    named = np.asarray(tiles)
    figure = go.Figure(
        [
            go.Scatter(
                x=projection[held == name, 0],
                y=projection[held == name, 1],
                mode="markers",
                name=name,
                text=named[held == name],
                hoverinfo="text+name",
            )
            for name in classes
        ]
    )
    run.log(dict(metrics) | {"class_distances": table, "umap": wandb.Plotly(figure)})
