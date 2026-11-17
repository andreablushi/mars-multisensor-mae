"""Evaluating one logged model: run here by default, or submitted with --dh."""

from __future__ import annotations

from dataclasses import asdict

import torch
from dhub.store import published_dataset_build
from dhub.submit import requested_read_workers, run_stage
from digitalhub_runtime_python import handler
from mars_multisensor_dataset.building.configs import sharad
from wandb.sdk.wandb_run import Run

from architecture.mae import CrossSensorMAE
from configs.load import load_config, named_generator
from configs.model import built_model
from configs.paths import results_directory
from evaluation.evaluate import evaluate
from evaluation.lavaflow import evaluate_lavaflow
from evaluation.reconstruct import evaluate_reconstruction
from logs.evaluation import (
    EVALUATION,
    LAVAFLOW,
    PART_STEPS,
    RECONSTRUCTION,
    kept_evaluation_values,
    log_kept_part,
    retire_evaluation_run,
)
from logs.tracker import (
    console_logger,
    logged_checkpoint,
    start_logging,
    tracked_run,
)
from training.step import selected_masking

log = console_logger(__name__)


def trained_model(model: CrossSensorMAE, run: Run, run_name: str) -> CrossSensorMAE:
    """Return the model with a logged run's latest weights, a "random" run's as drawn.

    Args:
        model: The model, as built.
        run: The tracked run that reads the weights.
        run_name: What the trained run is called, which names its model, a last
            part of "random" keeping the seed's own weights.

    Returns:
        model: The same model, its weights loaded.
    """
    if run_name.split("/")[-1] != "random":
        held = torch.load(logged_checkpoint(run, run_name), map_location="cpu")
        model.load_state_dict(held["model"])
        log.info("%s trained for %d steps", run_name, held["step"])
    return model


@handler()
def run_evaluation(
    run_name: str,
    overrides: list[str] | None = None,
    reconstruction: bool = False,
    lavaflow: bool = False,
) -> None:
    """Measure one logged model's latent space, rebuild some tiles, retrieve lava flows.

    Args:
        run_name: What the run is called, which names the model read and the
            tracked run, a last part of "random" measuring the seed's own weights.
        overrides: What to change in the configs, as dotted keys.
        reconstruction: Whether to rebuild and draw the tiles, every part when no
            part is asked.
        lavaflow: Whether to run the lava flow retrieval, every part when no part
            is asked; a part asked alone rebuilds the model's evaluation run, the
            other parts copied from the old one, which is then deleted.
    """
    config = load_config(overrides or [])
    whole = not (reconstruction or lavaflow)
    settings = config.evaluation.lavaflow
    name = sharad.LAYOUT.instrument
    retrieved = settings.build is not None and name in config.model.instruments
    if lavaflow and not retrieved:
        raise ValueError("the lava flow retrieval needs a build and SHARAD")
    build = published_dataset_build(config, config.evaluation.build)
    model, device, sizes, shapes = built_model(config, build)
    workers = requested_read_workers(config, EVALUATION)
    asked = [
        part
        for part, on in (
            (EVALUATION, whole),
            (RECONSTRUCTION, whole or reconstruction),
            (LAVAFLOW, (whole or lavaflow) and retrieved),
        )
        if on
    ]
    old = None if whole else tracked_run(run_name, EVALUATION)
    kept = (
        {}
        if old is None
        else kept_evaluation_values(old, asked, results_directory(run_name, "kept"))
    )
    run = start_logging(
        asdict(config),
        run_name,
        EVALUATION,
        {"device": str(device)},
        spared=old and old.id,
    )
    log.info("evaluating %s on %s", run_name, device)
    model = trained_model(model, run, run_name)
    if EVALUATION in kept:
        log_kept_part(run, *kept[EVALUATION], PART_STEPS[EVALUATION])
    if whole:
        rank_build = published_dataset_build(config, config.dataset.build)
        rank_tiles = [
            sorted(rank_build.read_observation_metadata())[at]
            for at in torch.randperm(
                len(rank_build.read_observation_metadata()),
                generator=named_generator(config.dataset.seed, "rankme"),
            )[: config.evaluation.rank_tiles].tolist()
        ]
        evaluate(
            model,
            build,
            sizes,
            shapes,
            config.evaluation.batch_size,
            config.evaluation.neighbours,
            rank_build,
            rank_tiles,
            config.dataset.budget,
            config.training.batch_size,
            config.dataset.seed,
            workers,
            device,
            run,
            run_name,
        )
    if RECONSTRUCTION in kept:
        log_kept_part(run, *kept[RECONSTRUCTION], PART_STEPS[RECONSTRUCTION])
    if RECONSTRUCTION in asked:
        ranging = config.evaluation.reconstruction.ranging
        evaluate_reconstruction(
            model,
            build,
            config.evaluation.reconstruction.tiles,
            sizes,
            shapes,
            selected_masking(
                config.training.masking.strategy,
                config.training.masking.cell_m,
                config.training.masking.independent,
            ),
            config.training.masking.mask_ratio,
            ranging if set(ranging) <= set(config.model.instruments) else [],
            config.dataset.seed,
            device,
            run,
            run_name,
        )
    if LAVAFLOW in kept:
        log_kept_part(run, *kept[LAVAFLOW], PART_STEPS[LAVAFLOW])
    if LAVAFLOW in asked:
        evaluate_lavaflow(
            model,
            published_dataset_build(config, settings.build),
            sizes[name],
            shapes[name],
            config.evaluation.batch_size,
            settings.qualitative.queries,
            settings.qualitative.top_k,
            settings.qualitative.pooling,
            config.dataset.seed,
            workers,
            device,
            run,
        )
    run.finish()
    if old is not None:
        retire_evaluation_run(old, asked)


if __name__ == "__main__":
    run_stage(
        EVALUATION,
        run_evaluation,
        __doc__,
        {
            "reconstruction": "rebuild and draw the tiles alone",
            "lavaflow": "run the lava flow retrieval alone",
        },
    )
