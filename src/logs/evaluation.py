"""Logging what one evaluation's files come to: scores, matrices, the rebuilt tiles."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import wandb
from wandb.sdk.wandb_run import Run

from evaluation.metrics import compute_metrics
from evaluation.results import read_tile_arrays, read_tile_vectors
from evaluation.similarity import cosine_similarity
from logs.attention import attention_image
from logs.reconstruction import PANELS, reconstruction_images
from logs.tracker import artifact_name

EVALUATION = "evaluation"
BASELINE = "random"
SCORES = f"{EVALUATION}/scores"
CONFUSION = f"{EVALUATION}/confusion"
DISTANCES = f"{EVALUATION}/distances"
RECONSTRUCTION = f"{EVALUATION}/reconstruction"
ATTENTION = f"{EVALUATION}/attention"
NEIGHBOURS = 5


def log_evaluation(
    run: Run,
    results: Path,
    reconstructions: Mapping[str, Path],
    attentions: Mapping[tuple[str, str], Path],
    run_name: str,
) -> None:
    """Log every score, both class matrices, the rebuilt tiles and where CMR looked.

    Args:
        run: The tracked run.
        results: The tiles' vectors, as write_tile_vectors wrote them.
        reconstructions: Each rebuilt tile, as write_tile_arrays wrote it, keyed
            by what it shows.
        attentions: Each tile's instrument rebuilt from the others whole, as
            write_tile_arrays wrote it, keyed by what the tile shows and the
            instrument.
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
    media = {}
    for place, path in reconstructions.items():
        for title, image in reconstruction_images(read_tile_arrays(path)).items():
            media[f"{RECONSTRUCTION}/{place}/{title}"] = wandb.Image(
                image, caption=" | ".join(PANELS)
            )
    for (place, query), path in attentions.items():
        image, caption = attention_image(read_tile_arrays(path), query)
        media[f"{ATTENTION}/{query}/{place}"] = wandb.Image(image, caption=caption)
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
            **media,
        }
    )
    files = wandb.Artifact(artifact_name(EVALUATION, run_name), type=EVALUATION)
    for path in (results, *reconstructions.values(), *attentions.values()):
        files.add_file(str(path))
    run.log_artifact(files)
