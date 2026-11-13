"""Training one run: run here by default, or submitted with --dh."""

from __future__ import annotations

import shutil
from dataclasses import asdict

from benchmark import run_benchmark
from dhub.store import published_dataset_build
from dhub.submit import requested_read_workers, run_stage
from digitalhub_runtime_python import handler
from evaluate import run_evaluation
from torch.utils.data import random_split

from configs.load import load_config, named_generator
from configs.model import built_model
from logs.tracker import TRAINING, console_logger, log_checkpoint, start_logging
from training.train import train

log = console_logger(__name__)


@handler()
def run_training(
    run_name: str,
    overrides: list[str] | None = None,
    evaluate: bool = False,
    benchmark: bool = False,
) -> None:
    """Train one run, log its last checkpoint, then evaluate or benchmark it if asked.

    Args:
        run_name: What the run is called, which names the model it saves and the
            tracked run it logs to, the part before a slash grouping it.
        overrides: What to change in the configs, as dotted keys.
        evaluate: Whether to evaluate the checkpoint once trained.
        benchmark: Whether to benchmark the checkpoint once trained.
    """
    config = load_config(overrides or [])
    build = published_dataset_build(config, config.dataset.build)
    model, device, sizes, shapes = built_model(config, build)
    workers = requested_read_workers(config, TRAINING)
    tiles = sorted(build.read_observation_metadata())
    training, validation = random_split(
        tiles, config.dataset.split, named_generator(config.dataset.seed, "split")
    )
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
        shapes,
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
    shutil.rmtree(build.root / "ready", ignore_errors=True)
    if evaluate:
        run_evaluation.__wrapped__(run_name, overrides)
    if benchmark:
        run_benchmark.__wrapped__(run_name, overrides)


if __name__ == "__main__":
    run_stage(
        TRAINING,
        run_training,
        __doc__,
        {
            "evaluate": "evaluate the model once trained",
            "benchmark": "benchmark the model once trained",
        },
    )
