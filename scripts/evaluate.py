"""Evaluating one logged model: run here by default, or submitted with --dh."""

from __future__ import annotations

from dataclasses import asdict

from dhub.store import published_build, published_dataset_build
from dhub.submit import requested_read_workers, run_stage
from digitalhub_runtime_python import handler
from mars_multisensor_dataset.building.configs import sharad
from utils.model import built_pretrained_model

from mars_multisensor_mae.architecture.load import cross_sensor_mae_with_logged_weights
from mars_multisensor_mae.architecture.models.context import Pooling
from mars_multisensor_mae.configs.load import load_config, named_sample
from mars_multisensor_mae.configs.paths import ARTIFACTS_ROOT, results_directory
from mars_multisensor_mae.evaluation.evaluate import evaluate
from mars_multisensor_mae.evaluation.lavaflow import evaluate_lavaflow
from mars_multisensor_mae.evaluation.reconstruct import evaluate_reconstruction
from mars_multisensor_mae.logs.evaluation import (
    EVALUATION,
    LAVAFLOW,
    PART_STEPS,
    RECONSTRUCTION,
    kept_evaluation_values,
    log_kept_part,
    retire_evaluation_run,
)
from mars_multisensor_mae.logs.tracker import console_logger, start_logging, tracked_run

log = console_logger(__name__)


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
    model, device, layout = built_pretrained_model(config, build)
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
    model = cross_sensor_mae_with_logged_weights(
        model, run, run_name, ARTIFACTS_ROOT
    ).eval()
    if EVALUATION in kept:
        log_kept_part(run, *kept[EVALUATION], PART_STEPS[EVALUATION])
    if whole:
        rank_build = published_dataset_build(config, config.dataset.build)
        rank_tiles = named_sample(
            sorted(rank_build.read_observation_metadata()),
            config.evaluation.rank_tiles,
            config.dataset.seed,
            "rankme",
        )
        evaluate(
            model,
            build,
            layout,
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
            layout,
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
            published_build(
                config, f"{config.dataset.version}-{settings.build}", "dataset"
            ),
            layout.only([name]),
            config.evaluation.batch_size,
            settings.qualitative.queries,
            settings.qualitative.top_k,
            Pooling(settings.qualitative.pooling),
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
