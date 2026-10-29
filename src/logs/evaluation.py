"""Logging what one evaluation comes to: scores, matrices, the rebuilt tiles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import wandb
from wandb.sdk.wandb_run import Run

from evaluation.metrics import compute_metrics, cosine_similarity
from logs.attention import attention_image, sounder_read_image
from logs.reconstruction import PANELS, SOUNDER, reconstruction_images
from logs.tokens import token_image
from logs.tracker import artifact_name

EVALUATION = "evaluation"
BASELINE = "random"
SCORES = f"{EVALUATION}/scores"
CONFUSION = f"{EVALUATION}/confusion"
DISTANCES = f"{EVALUATION}/distances"
RECONSTRUCTION = f"{EVALUATION}/reconstruction"
ATTENTION = f"{EVALUATION}/attention"
RANGING = f"{EVALUATION}/ranging"
TOKENS = f"{EVALUATION}/tokens"
NEIGHBOURS = 5


def log_evaluation(
    run: Run,
    run_name: str,
    vectors: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    classes: Mapping[str, str],
    reconstructions: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    attentions: Mapping[tuple[str, str], Mapping[str, Mapping[str, np.ndarray]]],
    rangings: Mapping[tuple[str, str], Mapping[str, Mapping[str, np.ndarray]]],
    files: Sequence[Path],
) -> None:
    """Log every score, both class matrices, the rebuilt tiles and where CMR looked.

    Args:
        run: The tracked run.
        run_name: What the run is called, which names its artifact.
        vectors: Each tile's vector, keyed by pooling, then view, then tile.
        classes: The class each tile earned, keyed by the tile.
        reconstructions: Each rebuilt tile's arrays, keyed by what it shows.
        attentions: Each tile's instrument rebuilt from the others whole, keyed by
            what the tile shows and the instrument.
        rangings: Each tile's ranging instrument rebuilt from the other alone, keyed
            alike.
        files: What the evaluation wrote, logged as its artifact.
    """
    measured = {}
    for pooling, by_view in vectors.items():
        for view, by_tile in by_view.items():
            tiles = sorted(by_tile)
            held = np.array([by_tile[tile] for tile in tiles], np.float64)
            measured[pooling, view] = compute_metrics(
                cosine_similarity(held), [classes[tile] for tile in tiles], NEIGHBOURS
            )
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
    for place, reconstruction in reconstructions.items():
        for title, image in reconstruction_images(reconstruction).items():
            media[f"{RECONSTRUCTION}/{place}/{title}"] = wandb.Image(
                image, caption=" | ".join(PANELS)
            )
        image, caption = token_image(reconstruction)
        media[f"{TOKENS}/{place}"] = wandb.Image(image, caption=caption)
    for (place, query), attention in attentions.items():
        image, caption = attention_image(attention, query)
        media[f"{ATTENTION}/{query}/{place}"] = wandb.Image(image, caption=caption)
    for (place, query), attention in rangings.items():
        drawn = attention_image if query == SOUNDER else sounder_read_image
        image, caption = drawn(attention, query)
        media[f"{RANGING}/{query}/{place}"] = wandb.Image(image, caption=caption)
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
    artifact = wandb.Artifact(artifact_name(EVALUATION, run_name), type=EVALUATION)
    for path in files:
        artifact.add_file(str(path))
    run.log_artifact(artifact)
