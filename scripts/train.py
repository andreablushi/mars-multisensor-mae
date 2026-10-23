"""Training one run: run here by default, or submitted with --dh."""

from __future__ import annotations

from dataclasses import asdict

from dhub.store import publish_checkpoint, published_build
from dhub.submit import run_stage, stage_workers
from digitalhub_runtime_python import handler
from evaluate import built_model, evaluate_checkpoint

from configs.load import load_config
from dataset.loader import (
    TRAINING_SPLIT,
    VALIDATION_SPLIT,
    fill_cache,
    loaders_by_split,
)
from logs.console import console_logger
from logs.tracker import start_logging
from training.train import train

TRAINING_STAGE = "training"
TRAINING_HANDLER = "scripts.train:run_training"

log = console_logger(__name__)


@handler(outputs=["model"])
def run_training(
    run_name: str,
    project=None,
    overrides: list[str] | None = None,
    evaluate: bool = False,
):
    """Train one run, publish the checkpoint it ended on, and evaluate it if asked.

    Args:
        run_name: What the run is called, which names the model it saves and the
            tracked run it logs to, the part before a slash grouping it.
        project: The DigitalHub project the model is logged into, or None here.
        overrides: What to change in the configs, as dotted keys.
        evaluate: Whether to evaluate the checkpoint once the training ends.

    Returns:
        model: The published checkpoint, or where it was written on a run here.
    """
    config = load_config(overrides or [])
    build = published_build(
        config.dataset.build,
        config.dataset.root,
        config.dataset.clip,
        config.model.clutter_conditioning,
    )
    model, device, sizes, shapes = built_model(config, build)
    workers = stage_workers(TRAINING_STAGE)
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
        TRAINING_STAGE,
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
    run.finish()
    published = last
    if project is not None:
        published = publish_checkpoint(project, last, run_name)
    if evaluate:
        evaluate_checkpoint(config, run_name, last, project)
    return published


if __name__ == "__main__":
    run_stage(
        TRAINING_STAGE,
        TRAINING_HANDLER,
        run_training,
        __doc__,
        {"evaluate": "evaluate the model once trained"},
    )
