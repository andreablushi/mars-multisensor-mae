"""Training one run's phases in order: run here by default, or submitted with --dh."""

from __future__ import annotations

from dataclasses import asdict

import torch
from dhub.store import publish_checkpoint, published_build, published_checkpoint
from dhub.submit import run_stage, stage_workers
from digitalhub_runtime_python import handler
from evaluate import built_model, evaluate_checkpoint

from configs.load import load_config
from dataset.loader import TRAINING_SPLIT, VALIDATION_SPLIT, loaders_by_split
from logs.console import console_logger
from logs.tracker import start_logging
from training.train import train

TRAINING_STAGE = "training"
TRAINING_HANDLER = "scripts.train:run_training"
PHASES = ("dem_delay", "surface", "shared")

log = console_logger(__name__)


@handler(outputs=["model"])
def run_training(
    project=None, overrides: list[str] | None = None, evaluate: bool = False
):
    """Train every phase in order, or the one named, publishing and evaluating each.

    Args:
        project: The DigitalHub project the models are logged into, or None here.
        overrides: What to compose the config with, as hydra spells them.
        evaluate: Whether to evaluate each phase's last checkpoint once it ends.

    Returns:
        model: The last phase's published checkpoint, or where it was written here.
    """
    overrides = overrides or []
    named = any(one.startswith("phase=") for one in overrides)
    last = None
    for phase in [[]] if named else [[f"phase={one}"] for one in PHASES]:
        config = load_config([*overrides, *phase])
        build = published_build(config.dataset.build, config.dataset.root)
        model, device, sizes, shapes = built_model(config, build)
        first_step = 0
        if config.training.init is not None:
            # A later phase holds instruments an earlier one never trained
            held = last or published_checkpoint(
                config.phase_model_name(config.training.init),
                config.training.checkpoints,
            )
            log.info("starting from %s", held)
            loaded = torch.load(held, map_location="cpu")
            model.load_state_dict(loaded["model"], strict=False)
            first_step = loaded["step"]
        loaders = loaders_by_split(
            build,
            sizes,
            config.dataset.pool,
            shapes,
            config.dataset.budget,
            config.dataset.budget_scale,
            config.dataset.split,
            config.dataset.seed,
            config.training.batch_size,
            stage_workers(TRAINING_STAGE),
        )
        training, validation = loaders[TRAINING_SPLIT], loaders[VALIDATION_SPLIT]
        log.info("training %s on %s", config.training.phase, device)
        run = start_logging(
            asdict(config),
            config.model_name,
            config.run_name,
            f"{TRAINING_STAGE}.{config.training.phase}",
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
            first_step,
            config.training.epochs,
            config.training.learning_rate,
            config.training.crossencoder_rate,
            config.training.weight_decay,
            config.training.warmup_epochs,
            config.training.validate_every,
            config.training.mask_ratio,
            config.training.cmr,
            config.training.frozen,
            config.training.checkpoints,
            config.dataset.seed,
            device,
            run,
        )
        run.finish()
        published = last
        if project is not None:
            published = publish_checkpoint(project, last, config.model_name)
        if evaluate:
            evaluate_checkpoint(config, last, project)
    return published


if __name__ == "__main__":
    run_stage(
        TRAINING_STAGE,
        TRAINING_HANDLER,
        run_training,
        __doc__,
        {"evaluate": "evaluate the model once trained"},
    )
