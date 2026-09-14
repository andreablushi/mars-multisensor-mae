"""Putting what one evaluation measured on the tracked run, numbers and figures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import wandb
from matplotlib import pyplot
from matplotlib.figure import Figure
from wandb.sdk.wandb_run import Run

from evaluation.evaluate import Evaluation


def class_similarity_figure(similarity: np.ndarray, names: Sequence[str]) -> Figure:
    """Return how alike each pair of classes is, drawn as a heatmap.

    Args:
        similarity: The mean cosine similarity of every ordered pair of classes. (C, C)
        names: The classes, in the order the matrix holds them.

    Returns:
        figure: The heatmap, each cell written with its own similarity.
    """
    side = 2 + len(names) * 0.6
    figure, axes = pyplot.subplots(figsize=(side, side))
    drawn = axes.imshow(similarity, cmap="viridis")
    axes.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
    axes.set_yticks(range(len(names)), names, fontsize=7)
    for row in range(len(names)):
        for column in range(len(names)):
            axes.text(
                column,
                row,
                f"{similarity[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white",
            )
    figure.colorbar(drawn, ax=axes, shrink=0.8)
    figure.tight_layout()
    return figure


def projection_figure(placed: np.ndarray, classes: Sequence[str]) -> Figure:
    """Return the latents laid out on the plane, coloured by the class of each.

    Args:
        placed: Where each latent sits on the plane. (N, 2)
        classes: The class of each of them, in that same order.

    Returns:
        figure: The scatter, one colour per class.
    """
    held = np.asarray(classes)
    figure, axes = pyplot.subplots(figsize=(8, 6))
    for one in sorted(set(classes)):
        inside = held == one
        axes.scatter(placed[inside, 0], placed[inside, 1], s=6, label=one)
    axes.legend(fontsize=7, markerscale=2, loc="best")
    axes.set_xticks([])
    axes.set_yticks([])
    figure.tight_layout()
    return figure


def metrics_figure(
    metrics: Mapping[str, float], intervals: Mapping[str, float]
) -> Figure:
    """Return every measured mean drawn as a bar, with the interval around it.

    Args:
        metrics: Every measured number, keyed as it is logged.
        intervals: Half the width of the 95% interval around each.

    Returns:
        figure: The bars, one per mean, each written with its own value.
    """
    shown = [name for name, half in intervals.items() if half > 0]
    figure, axes = pyplot.subplots(figsize=(2 + len(shown) * 0.7, 4))
    values = [metrics[name] for name in shown]
    error = [intervals[name] for name in shown]
    axes.bar(range(len(shown)), values, yerr=error, capsize=4, color="#4c72b0")
    axes.set_xticks(range(len(shown)), shown, rotation=90, fontsize=7)
    axes.axhline(0, color="black", linewidth=0.6)
    for at, (value, half) in enumerate(zip(values, error, strict=True)):
        axes.text(
            at, value + half, f"{value:.2f}", ha="center", va="bottom", fontsize=6
        )
    axes.set_ylabel("mean, 95% interval", fontsize=8)
    figure.tight_layout()
    return figure


def report_latent_space(run: Run, evaluation: Evaluation) -> None:
    """Log what one evaluation made of the latent space, its numbers and its figures.

    Args:
        run: The tracked run.
        evaluation: What the pass over the split made of its latent space.
    """
    similarity = class_similarity_figure(evaluation.similarity, evaluation.names)
    projection = projection_figure(evaluation.placed, evaluation.classes)
    measured = metrics_figure(evaluation.metrics, evaluation.intervals)
    run.log(
        {f"latent/{name}": value for name, value in evaluation.metrics.items()}
        | {
            f"latent/{name}_interval": half
            for name, half in evaluation.intervals.items()
        }
        | {
            "latent/class_similarity": wandb.Image(similarity),
            "latent/projection": wandb.Image(projection),
            "latent/metrics": wandb.Image(measured),
        }
    )
    for figure in (similarity, projection, measured):
        pyplot.close(figure)


def report_reconstruction(run: Run, metrics: Mapping[str, float]) -> None:
    """Log what one evaluation made of the reconstruction.

    Args:
        run: The tracked run.
        metrics: What the masked pass rebuilt, keyed as the evaluation names it.
    """
    run.log({f"recon_{name}": value for name, value in metrics.items()})
