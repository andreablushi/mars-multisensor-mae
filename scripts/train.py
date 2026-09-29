"""Training one run: run here by default, or submitted with --dh."""

from __future__ import annotations

from dataclasses import asdict
from functools import partial

from dhub.configs import load_platform, stage_workers
from dhub.publish import publish_checkpoint, published_name
from dhub.store import published_build
from dhub.submit import ran_stage
from digitalhub_runtime_python import handler
from evaluate import built_model, evaluate_checkpoint

from architecture.tokens import token_batch_padding
from configs.load import load_config
from dataset.loader import TRAINING_SPLIT, VALIDATION_SPLIT, loaders_by_split
from logs.console import console_logger
from logs.tracker import start_logging
from training.train import train

TRAINING_STAGE = "training"
TRAINING_HANDLER = "scripts.train:run_training"

_MODEL = load_platform().publishes["model"]

log = console_logger(__name__)


@handler(outputs=[_MODEL])
def run_training(
    project=None, overrides: list[str] | None = None, evaluate: bool = False
):
    """Train one run, publish the best checkpoint it left, and evaluate it if asked.

    Args:
        project: The DigitalHub project the model is logged into, or None here.
        overrides: What to compose the config with, as hydra spells them.
        evaluate: Whether to evaluate the best checkpoint once the training ends.

    Returns:
        model: The published checkpoint, or where it was written on a run here.
    """
    config = load_config(overrides or [])
    build = published_build(config.dataset.build, config.dataset.root)
    model, device, sizes, shapes, strides = built_model(config, build)
    loaders = loaders_by_split(
        build,
        sizes,
        config.dataset.pool,
        shapes,
        partial(
            token_batch_padding,
            cell_m=config.model.cell_m,
            delay_rows=config.dataset.patchsize["SHARAD"]["delay"],
        ),
        config.dataset.split,
        config.dataset.seed,
        config.model.delay,
        config.dataset.scaled_per_observation,
        config.training.batch_size,
        stage_workers(TRAINING_STAGE),
    )
    training, validation = loaders[TRAINING_SPLIT], loaders[VALIDATION_SPLIT]
    log.info("training on %s", device)
    run = start_logging(
        asdict(config),
        config.run_name,
        TRAINING_STAGE,
        {
            "device": str(device),
            "parameters": sum(one.numel() for one in model.parameters()),
            "shapes": shapes,
            "strides": strides,
            "tiles": {
                "training": len(training.dataset),
                "validation": len(validation.dataset),
            },
        },
    )
    best = train(
        model,
        training,
        validation,
        config.training.max_steps,
        config.training.learning_rate,
        config.training.weight_decay,
        config.training.warmup_steps,
        config.training.validate_every,
        config.training.patience,
        config.training.mask_ratio,
        config.training.checkpoints,
        config.dataset.seed,
        device,
        run,
    )
    run.finish()
    published = best
    if project is not None:
        name = published_name("model", config.run_name)
        published = publish_checkpoint(project, best, name)
    if evaluate:
        evaluate_checkpoint(config, best, project)
    return published


if __name__ == "__main__":
    raise SystemExit(
        ran_stage(
            TRAINING_STAGE,
            TRAINING_HANDLER,
            run_training,
            __doc__,
            {"evaluate": "evaluate the model once trained"},
        )
    )
