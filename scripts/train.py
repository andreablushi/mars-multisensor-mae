"""Training one run: run here by default, or submitted with --dh."""

from __future__ import annotations

from dataclasses import asdict

from dhub.store import published_build
from dhub.submit import run_stage, stage_workers
from digitalhub_runtime_python import handler
from evaluate import CLUTTER_NONE, built_model, evaluate_checkpoint

from configs.load import load_config
from dataset.loader import (
    TRAINING_SPLIT,
    VALIDATION_SPLIT,
    fill_cache,
    loaders_by_split,
)
from logs.console import console_logger
from logs.tracker import TRAINING, log_checkpoint, start_logging
from training.train import train

TRAINING_HANDLER = "scripts.train:run_training"

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
    build = published_build(
        config.dataset.build,
        config.dataset.root,
        config.dataset.clip,
        config.model.clutter != CLUTTER_NONE,
    )
    model, device, sizes, shapes = built_model(config, build)
    workers = stage_workers(TRAINING)
    loaders = loaders_by_split(
        build,
        sizes,
        shapes,
        config.dataset.budget,
        config.dataset.budget_scale,
        config.dataset.split,
        config.dataset.seed,
        config.training.batch_size,
        workers,
    )
    training, validation = loaders[TRAINING_SPLIT], loaders[VALIDATION_SPLIT]
    contrastive = config.training.contrastive
    fill_cache(build, sizes, shapes, workers)
    log.info("training on %s", device)
    run = start_logging(
        asdict(config),
        run_name,
        TRAINING,
        {
            "device": str(device),
            "parameters": sum(one.numel() for one in model.parameters()),
            "shapes": shapes,
            "tiles": {
                "training": len(training.dataset),
                "validation": len(validation.dataset),
            },
        },
    )
    last = train(
        model,
        training,
        validation,
        config.training.num_steps,
        config.training.learning_rate,
        config.training.weight_decay,
        config.training.warmup_steps,
        config.training.validate_every,
        config.training.mask_ratio,
        contrastive.instruments if contrastive.enabled else [],
        contrastive.pooling,
        contrastive.temperature,
        config.training.checkpoints,
        config.dataset.seed,
        device,
        run,
        asdict(config),
    )
    log_checkpoint(run, last, run_name)
    run.finish()
    if evaluate:
        evaluate_checkpoint(config, run_name)


if __name__ == "__main__":
    run_stage(
        TRAINING,
        TRAINING_HANDLER,
        run_training,
        __doc__,
        {"evaluate": "evaluate the model once trained"},
    )
