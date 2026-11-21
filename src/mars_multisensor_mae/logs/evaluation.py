"""Logging what one evaluation comes to: scores, class distances, the drawn tiles."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import wandb
from wandb.apis.public import Run as TrackedRun
from wandb.sdk.wandb_run import Run

from mars_multisensor_mae.evaluation.models.metrics import Metrics, Score
from mars_multisensor_mae.evaluation.models.vectors import TileVectors
from mars_multisensor_mae.logs.images.attention import attention_image
from mars_multisensor_mae.logs.images.modality_gap import modality_gap_image
from mars_multisensor_mae.logs.images.panels import DrawnImage
from mars_multisensor_mae.logs.images.ranging import ranging_image
from mars_multisensor_mae.logs.tracker import RESULTS_STEP, artifact_name

EVALUATION = "evaluation"

RECONSTRUCTION = "reconstruction"

LAVAFLOW = "lavaflow"

PART_STEPS = {EVALUATION: 0, RECONSTRUCTION: 0, LAVAFLOW: RESULTS_STEP}


def log_evaluation(
    run: Run,
    run_name: str,
    metrics: Sequence[Metrics],
    gaps: Sequence[Score],
    vectors: Sequence[TileVectors],
    directory: Path,
) -> None:
    """Log the scores, class distances and modality gap, the vectors as an artifact.

    Args:
        run: The tracked run.
        run_name: What the run is called, which names its artifact.
        metrics: Each pooling and view's metrics.
        gaps: Each pooling and pair of instruments' RMG.
        vectors: Each pooling's unit tile vectors.
        directory: Where the evaluation wrote its vectors, logged as its artifact.
    """
    scores = [score for one in metrics for score in one.scores()] + list(gaps)
    drawn = {
        pooled.pooling: modality_gap_image(pooled.instruments) for pooled in vectors
    }
    run.summary.update(score_summary(EVALUATION, scores))
    run.log(
        {
            f"{EVALUATION}/scores": wandb.Table(
                columns=list(Score._fields), data=[list(score) for score in scores]
            ),
            f"{EVALUATION}/distances": wandb.Table(
                columns=["pooling", "view", "row", "column", "value"],
                data=[
                    [one.pooling, one.view, row, column, float(value)]
                    for one in metrics
                    for row, held in zip(one.classes, one.distances, strict=True)
                    for column, value in zip(one.classes, held, strict=True)
                ],
            ),
            **{
                f"{EVALUATION}/modality_gap/{pooling}": wandb.Image(
                    image, caption=caption
                )
                for pooling, (image, caption) in drawn.items()
            },
        },
        step=PART_STEPS[EVALUATION],
    )
    artifact = wandb.Artifact(artifact_name(EVALUATION, run_name), type=EVALUATION)
    artifact.add_dir(str(directory))
    run.log_artifact(artifact)


def log_reconstruction(
    run: Run,
    run_name: str,
    reconstructions: Mapping[tuple[str, str], Mapping[str, Mapping[str, np.ndarray]]],
    rangings: Mapping[tuple[str, str], Mapping[str, Mapping[str, np.ndarray]]],
    directory: Path,
) -> None:
    """Log the rebuilt tiles and where CMR looked, their arrays as the run's artifact.

    Args:
        run: The tracked run.
        run_name: What the run is called, which names its artifact.
        reconstructions: Each tile's instrument masked and rebuilt from the others
            whole, keyed by what the tile shows and the instrument.
        rangings: Each tile's ranging instrument rebuilt from the other alone, keyed
            alike.
        directory: Where the drawn tiles were written, logged as its artifact.
    """
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
    run.log(media, step=PART_STEPS[RECONSTRUCTION])
    artifact = wandb.Artifact(
        artifact_name(RECONSTRUCTION, run_name), type=RECONSTRUCTION
    )
    artifact.add_dir(str(directory))
    run.log_artifact(artifact)


def log_lavaflow(
    run: Run,
    scores: Sequence[Score],
    images: Sequence[DrawnImage],
) -> None:
    """Log the lava flow retrieval's mAP and the drawn queries beside what they found.

    Args:
        run: The tracked run.
        scores: The reflector queries' mAP per pooling, its view the crops the
            gallery leaves out.
        images: Each drawn query's image and caption, logged as qualitative 1 on.
    """
    run.summary.update(score_summary(LAVAFLOW, scores))
    run.log(
        {
            f"{LAVAFLOW}/scores": wandb.Table(
                columns=list(Score._fields), data=[list(score) for score in scores]
            ),
            **{
                f"{LAVAFLOW}/qualitative-{at}": wandb.Image(image, caption=caption)
                for at, (image, caption) in enumerate(images, 1)
            },
        },
        step=PART_STEPS[LAVAFLOW],
    )


def score_summary(section: str, scores: Sequence[Score]) -> dict[str, float]:
    """Return some scores as run summary values.

    Args:
        section: The section the keys sit under, such as "evaluation".
        scores: The scores.

    Returns:
        summary: Each value keyed "<section>/<pooling>/<view>/<metric>".
    """
    return {
        f"{section}/{score.pooling}/{score.view}/{score.metric}": score.value
        for score in scores
    }


def evaluation_part(key: str) -> str:
    """Return which part of an evaluation logs a key of its run.

    Args:
        key: A history or summary key of an evaluation run.

    Returns:
        part: EVALUATION, RECONSTRUCTION or LAVAFLOW.
    """
    if key.startswith(f"{LAVAFLOW}/"):
        return LAVAFLOW
    if key.startswith((f"{EVALUATION}/reconstruction/", f"{EVALUATION}/ranging/")):
        return RECONSTRUCTION
    return EVALUATION


def kept_evaluation_values(
    old: TrackedRun, asked: Sequence[str], directory: Path
) -> dict[str, tuple[dict[str, object], dict[str, object]]]:
    """Return what every part not asked logged in an evaluation run, ready to log again.

    Args:
        old: The evaluation run whose parts are kept.
        asked: The parts logged anew, left out.
        directory: Where the kept images and tables are downloaded, emptied first.

    Returns:
        kept: Each kept part's last value of every history key and its summary
            scalars, keyed by part.
    """
    shutil.rmtree(directory, ignore_errors=True)
    history = {}
    for row in old.scan_history():
        history |= {
            key: value
            for key, value in row.items()
            if not key.startswith("_") and value is not None
        }
    kept = {part: ({}, {}) for part in PART_STEPS if part not in asked}
    for key, value in history.items():
        part = evaluation_part(key)
        if part not in kept:
            continue
        if isinstance(value, dict) and "path" in value:
            with old.file(value["path"]).download(str(directory), replace=True) as file:
                path = file.name
            value = (
                wandb.Table(**json.loads(Path(path).read_text()))
                if value.get("_type") == "table-file"
                else wandb.Image(path, caption=value.get("caption"))
            )
        kept[part][0][key] = value
    for key, value in old.summary.items():
        part = evaluation_part(key)
        if (
            part in kept
            and key not in history
            and not key.startswith("_")
            and isinstance(value, (int, float, str))
        ):
            kept[part][1][key] = value
    return kept


def log_kept_part(
    run: Run, history: Mapping[str, object], summary: Mapping[str, object], step: int
) -> None:
    """Log one kept part of an evaluation again at its own step.

    Args:
        run: The tracked run.
        history: The part's last value of every history key.
        summary: The part's summary scalars.
        step: The step the part logs at.
    """
    run.summary.update(dict(summary))
    if history:
        run.log(dict(history), step=step)


def retire_evaluation_run(old: TrackedRun, asked: Sequence[str]) -> None:
    """Delete a replaced evaluation run, keeping the artifacts of the parts not asked.

    Args:
        old: The evaluation run replaced.
        asked: The parts logged anew, whose old artifacts are deleted too unless
            still the latest, as when the new content matched and W&B kept them.
    """
    for artifact in old.logged_artifacts():
        if artifact.type in asked and "latest" not in artifact.aliases:
            artifact.delete(delete_aliases=True)
    old.delete(delete_artifacts=False)
