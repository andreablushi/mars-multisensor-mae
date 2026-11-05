"""Logging what one evaluation comes to: scores, class distances, the drawn tiles."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import wandb
from wandb.sdk.wandb_run import Run

from evaluation.metrics import Metrics
from logs.images.attention import attention_image
from logs.images.ranging import ranging_image
from logs.tracker import artifact_name

EVALUATION = "evaluation"


def log_evaluation(
    run: Run,
    run_name: str,
    metrics: Mapping[tuple[str, str], Metrics],
    reconstructions: Mapping[tuple[str, str], Mapping[str, Mapping[str, np.ndarray]]],
    rangings: Mapping[tuple[str, str], Mapping[str, Mapping[str, np.ndarray]]],
    directory: Path,
) -> None:
    """Log every score, the class distances, the rebuilt tiles and where CMR looked.

    Args:
        run: The tracked run.
        run_name: What the run is called, which names its artifact.
        metrics: Each view's metrics, keyed by its pooling and view.
        reconstructions: Each tile's instrument masked and rebuilt from the others
            whole, keyed by what the tile shows and the instrument.
        rangings: Each tile's ranging instrument rebuilt from the other alone, keyed
            alike.
        directory: Where the evaluation wrote its files, logged as its artifact.
    """
    scores = [
        [pooling, view, name, value]
        for (pooling, view), one in metrics.items()
        for name, value in {
            "mAP": float(one.retrieval["mAP"].mean()),
            **one.knn,
            "silhouette": float(one.silhouette.mean()),
        }.items()
    ]
    run.summary.update(
        {
            f"{EVALUATION}/{pooling}/{view}/{name}": value
            for pooling, view, name, value in scores
        }
    )
    media = {}
    for section, drawn, by_query in (
        ("reconstruction", attention_image, reconstructions),
        ("ranging", ranging_image, rangings),
    ):
        for (place, query), attention in by_query.items():
            image, caption = drawn(attention, query)
            media[f"{EVALUATION}/{section}/{query}/{place}"] = wandb.Image(
                image, caption=caption
            )
    run.log(
        {
            f"{EVALUATION}/scores": wandb.Table(
                columns=["pooling", "view", "metric", "value"], data=scores
            ),
            f"{EVALUATION}/distances": wandb.Table(
                columns=["pooling", "view", "row", "column", "value"],
                data=[
                    [pooling, view, row, column, float(value)]
                    for (pooling, view), one in metrics.items()
                    for row, held in zip(one.classes, one.distances, strict=True)
                    for column, value in zip(one.classes, held, strict=True)
                ],
            ),
            **media,
        }
    )
    artifact = wandb.Artifact(artifact_name(EVALUATION, run_name), type=EVALUATION)
    artifact.add_dir(str(directory))
    run.log_artifact(artifact)
