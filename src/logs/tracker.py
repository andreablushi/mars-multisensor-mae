"""Tracking a run on Weights & Biases, every metric named in one place."""

from __future__ import annotations

import os
from collections.abc import Mapping

import plotly.graph_objects as go
import wandb
from dotenv import load_dotenv
from torch import Tensor
from wandb.sdk.wandb_run import Run

from configs.paths import ENV_PATH
from figures.style import TEMPLATE


def start_logging(
    config: Mapping[str, object],
    name: str,
    group: str,
    stage: str,
    facts: Mapping[str, object],
) -> Run:
    """Return the tracked run of one stage, replacing any of its name and stage.

    Args:
        config: What the run reads, trains and how, which the run is recorded with.
        name: What the tracked run is called.
        group: What it is grouped under, so every phase of one run sits together.
        stage: What the stage is called, which tells one phase from another.
        facts: What only the assembled run knows, recorded beside the config.

    Returns:
        run: The run, named and grouped, its place read from the .env.
    """
    load_dotenv(ENV_PATH)
    for tracked in wandb.Api().runs(
        f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}",
        filters={"display_name": name},
    ):
        if tracked.job_type == stage:
            tracked.delete()
    run = wandb.init(config=dict(config), name=name, group=group, job_type=stage)
    run.config.update(dict(facts))
    return run


def log_step(run: Run, step: int, terms: Mapping[str, Tensor]) -> None:
    """Log one training step.

    Args:
        run: The tracked run.
        step: Which step of the whole run it is.
        terms: Every loss term, keyed as the loss names them, and the gradient norm.
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


def log_phases(
    run: Run,
    curves: Mapping[str, list[tuple[int, float]]],
    starts: Mapping[str, int],
    step: int,
) -> None:
    """Log every loss curve in one figure, a dashed line where each phase starts.

    Args:
        run: The tracked run.
        curves: Each curve so far, keyed as it is logged, as (step, value).
        starts: The step each phase starts after, keyed by phase.
        step: Which step of the whole run it is logged at.
    """
    figure = go.Figure(
        [
            go.Scatter(
                x=[at for at, _ in points], y=[one for _, one in points], name=name
            )
            for name, points in curves.items()
        ],
        layout={"template": TEMPLATE, "xaxis_title": "step", "yaxis_title": "loss"},
    )
    for phase, start in starts.items():
        figure.add_vline(x=start, line_dash="dash", annotation_text=phase)
    run.log({"phases": figure}, step=step)


def log_summary(run: Run, summary: Mapping[str, object]) -> None:
    """Write what the whole run came to.

    Args:
        run: The tracked run.
        summary: What it came to, keyed as it is recorded.
    """
    run.summary.update(dict(summary))
