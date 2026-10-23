"""Logging what one evaluation's files come to: scores, matrices, the rebuilt tiles."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import wandb
from wandb.sdk.wandb_run import Run

from evaluation.metrics import compute_metrics
from evaluation.results import read_reconstruction, read_tile_vectors
from evaluation.similarity import cosine_similarity
from logs.reconstruction import PANELS, reconstruction_images
from logs.tracker import artifact_name

EVALUATION = "evaluation"
BASELINE = "random"
SCORES = f"{EVALUATION}/scores"
CONFUSION = f"{EVALUATION}/confusion"
DISTANCES = f"{EVALUATION}/distances"
RECONSTRUCTION = f"{EVALUATION}/reconstruction"
NEIGHBOURS = 5


def log_evaluation(
    run: Run, results: Path, reconstructions: Mapping[str, Path], run_name: str
) -> None:
    """Log every score, both class matrices and the rebuilt tiles, and keep the files.

    Args:
        run: The tracked run.
        results: The tiles' vectors, as write_tile_vectors wrote them.
        reconstructions: Each rebuilt tile, as write_reconstruction wrote it, keyed
            by what it shows.
        run_name: What the run is called, which names its artifact.
    """
    _, labels, vectors = read_tile_vectors(results)
    measured = {
        view: compute_metrics(cosine_similarity(held), labels, NEIGHBOURS)
        for view, held in vectors.items()
    }
    scores = [
        [pooling, view, name, value]
        for (pooling, view), one in measured.items()
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
    run.log(
        {
            SCORES: wandb.Table(
                columns=["pooling", "view", "metric", "value"], data=scores
            ),
            **{
                key: wandb.Table(
                    columns=["pooling", "view", "row", "column", "value"],
                    data=[
                        [pooling, view, row, column, float(value)]
                        for (pooling, view), one in measured.items()
                        for row, held in zip(
                            one.classes, getattr(one, field), strict=True
                        )
                        for column, value in zip(one.classes, held, strict=True)
                    ],
                )
                for key, field in ((CONFUSION, "confusion"), (DISTANCES, "distances"))
            },
            **{
                f"{RECONSTRUCTION}/{place}/{title}": wandb.Image(
                    image, caption=" | ".join(PANELS)
                )
                for place, path in reconstructions.items()
                for title, image in reconstruction_images(
                    read_reconstruction(path)[1]
                ).items()
            },
        }
    )
    files = wandb.Artifact(artifact_name(EVALUATION, run_name), type=EVALUATION)
    for path in (results, *reconstructions.values()):
        files.add_file(str(path))
    run.log_artifact(files)
