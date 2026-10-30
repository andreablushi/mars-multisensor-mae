"""Training one run: run here by default, or submitted with --dh."""

from __future__ import annotations

from dataclasses import asdict

import torch
from dhub.store import published_dataset_build
from dhub.submit import requested_read_workers, run_stage
from digitalhub_runtime_python import handler
from evaluate import run_evaluation
from torch.utils.data import random_split

from configs.load import load_config, named_seed
from configs.model import built_model
from dataset.loader import cache_tiles
from logs.console import console_logger
from logs.tracker import TRAINING, log_checkpoint, start_logging
from training.train import train

log = console_logger(__name__)


@handler()
def run_training(
    run_name: str, overrides: list[str] | None = None, evaluate: bool = False
) -> None:
    """Train one run, log the checkpoint it ended on, and evaluate it if asked.

    Args:
        run_name: What the run is called, which names the model it saves and the
            tracked run it logs to, the part before a slash grouping it.
        overrides: What to change in the configs, as dotted keys.
        evaluate: Whether to evaluate the checkpoint once the training ends.
    """
    config = load_config(overrides or [])
    build = published_dataset_build(config, config.dataset.build)
    model, device, sizes, shapes = built_model(config, build)
    workers = requested_read_workers(TRAINING)
    # Cache the tiles to avoid reading them from disk every epoch
    cache_tiles(build, sizes, shapes, workers)
    tiles = sorted(build.read_observation_metadata())
    generator = torch.Generator().manual_seed(named_seed(config.dataset.seed, "split"))
    training, validation = random_split(tiles, config.dataset.split, generator)
    log.info("training on %s", device)
    run = start_logging(
        asdict(config),
        run_name,
        TRAINING,
        {
            "device": str(device),
            "parameters": sum(one.numel() for one in model.parameters()),
            "shapes": shapes,
            "tiles": {"training": len(training), "validation": len(validation)},
        },
    )
    last = train(
        model,
        build,
        training,
        validation,
        sizes,
        config.dataset.budget,
        config.dataset.budget_scale,
        config.training.batch_size,
        workers,
        config.training.num_steps,
        config.training.learning_rate,
        config.training.weight_decay,
        config.training.warmup_steps,
        config.training.validate_every,
        config.training.mask_ratio,
        config.training.contrastive.instruments,
        config.training.contrastive.pooling,
        config.training.contrastive.temperature,
        config.training.reconstruction_error,
        config.training.checkpoints,
        config.dataset.seed,
        device,
        run,
        asdict(config),
    )
    log_checkpoint(run, last, run_name)
    run.finish()
    if evaluate:
        run_evaluation.__wrapped__(run_name, overrides)


if __name__ == "__main__":
    run_stage(
        TRAINING, run_training, __doc__, {"evaluate": "evaluate the model once trained"}
    )
