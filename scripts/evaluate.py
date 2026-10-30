"""Evaluating one logged model: run here by default, or submitted with --dh."""

from __future__ import annotations

from dataclasses import asdict

import torch
from dhub.store import published_dataset_build
from dhub.submit import requested_read_workers, run_stage
from digitalhub_runtime_python import handler

from configs.load import load_config
from configs.model import built_model
from dataset.loader import cache_tiles
from evaluation.evaluate import evaluate
from logs.console import console_logger
from logs.evaluation import BASELINE, EVALUATION
from logs.tracker import logged_checkpoint, start_logging

log = console_logger(__name__)


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
    cache_tiles(build, sizes, shapes, workers)
    run = start_logging(asdict(config), run_name, EVALUATION, {"device": str(device)})
    if run_name.split("/")[-1] != BASELINE:
        held = torch.load(logged_checkpoint(run, run_name), map_location="cpu")
        model.load_state_dict(held["model"])
        log.info("%s trained for %d steps", run_name, held["step"])
    log.info("evaluating %s on %s", run_name, device)
    evaluate(
        model,
        build,
        sizes,
        shapes,
        config.evaluation.tiles,
        config.training.mask_ratio,
        config.evaluation.ranging,
        config.dataset.seed,
        workers,
        device,
        run,
        run_name,
    )
    run.finish()


if __name__ == "__main__":
    run_stage(EVALUATION, run_evaluation, __doc__)
