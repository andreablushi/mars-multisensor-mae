"""Putting what one evaluation measured on the tracked run, numbers and figures."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import wandb
from matplotlib import pyplot
from matplotlib.figure import Figure
from wandb.sdk.wandb_run import Run

from evaluation.evaluate import Evaluation


def class_similarity_figure(similarity: np.ndarray, names: Sequence[str]) -> Figure:
    """Return how alike each pair of classes is, drawn as a heatmap.

    Args:
        similarity: The mean cosine similarity of every ordered pair of
            classes. (C, C)
        names: The classes, in the order the matrix holds them.

    Returns:
        figure: The heatmap, each cell written with its own similarity, the
            diagonal holding a class against itself.
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


def report_evaluation(run: Run, evaluation: Evaluation) -> None:
    """Log what one evaluation measured, its numbers and its two figures.

    Args:
        run: The tracked run.
        evaluation: What the pass over the split made of its latent space.
    """
    similarity = class_similarity_figure(evaluation.similarity, evaluation.names)
    projection = projection_figure(evaluation.placed, evaluation.classes)
    run.log(
        {f"latent/{name}": value for name, value in evaluation.metrics.items()}
        | {
            "latent/class_similarity": wandb.Image(similarity),
            "latent/projection": wandb.Image(projection),
        }
    )
    pyplot.close(similarity)
    pyplot.close(projection)
