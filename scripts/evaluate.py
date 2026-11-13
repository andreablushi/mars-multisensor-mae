"""Evaluating one logged model: run here by default, or submitted with --dh."""

from __future__ import annotations

from dataclasses import asdict

import torch
from dhub.store import published_dataset_build
from dhub.submit import requested_read_workers, run_stage
from digitalhub_runtime_python import handler
from wandb.sdk.wandb_run import Run

from architecture.mae import CrossSensorMAE
from configs.load import load_config, named_generator
from configs.model import built_model
from evaluation.evaluate import evaluate
from logs.evaluation import EVALUATION
from logs.tracker import console_logger, logged_checkpoint, start_logging

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
def run_evaluation(run_name: str, overrides: list[str] | None = None) -> None:
    """Measure one logged model's latent space over the labelled tiles, rebuild some.

    Args:
        run_name: What the run is called, which names the model read and the
            tracked run, a last part of "random" measuring the seed's own weights.
        overrides: What to change in the configs, as dotted keys.
    """
    config = load_config(overrides or [])
    build = published_dataset_build(config, config.evaluation.build)
    model, device, sizes, shapes = built_model(config, build)
    workers = requested_read_workers(EVALUATION)
    rank_build = published_dataset_build(config, config.dataset.build)
    rank_tiles = [
        sorted(rank_build.read_observation_metadata())[at]
        for at in torch.randperm(
            len(rank_build.read_observation_metadata()),
            generator=named_generator(config.dataset.seed, "rankme"),
        )[: config.evaluation.rank_tiles].tolist()
    ]
    run = start_logging(asdict(config), run_name, EVALUATION, {"device": str(device)})
    log.info("evaluating %s on %s", run_name, device)
    evaluate(
        trained_model(model, run, run_name),
        build,
        sizes,
        shapes,
        config.evaluation.batch_size,
        config.evaluation.neighbours,
        rank_build,
        rank_tiles,
        config.dataset.budget,
        config.training.batch_size,
        config.evaluation.tiles,
        config.training.mask_ratio,
        config.evaluation.ranging
        if set(config.evaluation.ranging) <= set(config.model.instruments)
        else [],
        config.dataset.seed,
        workers,
        device,
        run,
        run_name,
    )
    run.finish()


if __name__ == "__main__":
    run_stage(EVALUATION, run_evaluation, __doc__)
