"""Benchmarking one logged model on SWIM's ice: run here by default, or with --dh."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch
from dhub.store import published_dataset_build, published_ice_labels
from dhub.submit import requested_read_workers, run_stage
from digitalhub_runtime_python import handler
from evaluate import trained_model
from wandb.sdk.wandb_run import Run

from architecture.mae import CrossSensorMAE
from benchmarking.evaluation import evaluate as evaluation
from benchmarking.finetune import finetune as finetuning
from configs.load import load_config, named_generator
from configs.model import built_benchmark_model, built_model
from configs.paths import heads_path
from dataset.loader import swim_targets_loader, tiles_split_by_square
from logs.benchmark import BENCHMARK, FINETUNE, HEADS
from logs.tracker import (
    console_logger,
    log_checkpoint,
    logged_checkpoint,
    start_logging,
)

log = console_logger(__name__)


@handler()
def run_benchmark(
    run_name: str,
    overrides: list[str] | None = None,
    finetune: bool = False,
    evaluate: bool = False,
) -> None:
    """Fine-tune and evaluate both protocols over one logged model, or one of the two.

    Args:
        run_name: What the run is called, which names the model read and the
            tracked runs, a last part of "random" benchmarking the seed's own weights.
        overrides: What to change in the configs, as dotted keys.
        finetune: Whether to fine-tune and log the heads, both phases when neither
            is asked.
        evaluate: Whether to evaluate the logged heads on the test tiles.
    """
    config = load_config(overrides or [])
    build = published_dataset_build(config, config.benchmark.build)
    model, device, sizes, shapes = built_model(config, build)
    workers = requested_read_workers(config, BENCHMARK)
    left = [
        one
        for one in config.model.instruments
        if one not in config.benchmark.instruments
    ]
    tracked = run_name + "".join(f"-no-{one.lower()}" for one in left)
    # Split the labelled tiles of the build by square
    by_tile = build.read_observation_metadata()
    labels = {
        tile: one
        for tile, one in published_ice_labels(config).items()
        if tile in by_tile
    }
    training, validation, test = tiles_split_by_square(
        labels, config.benchmark.folds, config.benchmark.block_deg, config.dataset.seed
    )
    targets = swim_targets_loader(labels)
    spread = np.stack([targets[tile] for tile in training]).std(axis=(0, 1))  # (Z,)
    settings = config.benchmark.finetune
    # Each share is the first tiles of one draw, so a larger share holds the smaller
    order = torch.randperm(
        len(training),
        generator=named_generator(config.dataset.seed, "benchmark/fraction"),
    ).tolist()
    shares = {
        f"{round(100 * fraction)}pct": [
            training[at] for at in order[: round(fraction * len(training))]
        ]
        for fraction in sorted(config.benchmark.fractions)
    }
    share_steps = dict(zip(shares, settings.fraction_steps, strict=True))
    pass_steps = len(training) // settings.batch_size
    # Every share is validated on the same first tiles of one draw
    checked = [
        validation[at]
        for at in torch.randperm(
            len(validation),
            generator=named_generator(config.dataset.seed, "benchmark/validation"),
        ).tolist()[: config.benchmark.validation_tiles]
    ]

    def tracked_run(stage: str) -> tuple[Run, CrossSensorMAE]:
        """Return one phase's tracked run, and the MAE it reads."""
        run = start_logging(asdict(config), tracked, stage, {"device": str(device)})
        log.info("%s of %s on %s", stage, tracked, device)
        return run, trained_model(model, run, run_name)

    # Train both protocols on each share of the training tiles, kept at their best
    if finetune or not evaluate:
        run, mae = tracked_run(FINETUNE)
        heads = {}
        for share, tiles in shares.items():
            whole = len(tiles) == len(training)
            num_steps = settings.epochs * pass_steps if whole else share_steps[share]
            heads[share] = finetuning.finetune(
                built_benchmark_model(config, mae, device),
                build,
                tiles,
                checked,
                targets,
                spread,
                {name: sizes[name] for name in config.benchmark.instruments},
                shapes,
                config.dataset.budget,
                config.dataset.budget_scale,
                settings.batch_size,
                config.benchmark.batch_size,
                workers,
                num_steps,
                round(num_steps * settings.warmup_epochs / settings.epochs),
                pass_steps if whole else num_steps // settings.validations,
                settings.head_learning_rate,
                settings.encoder_learning_rate,
                settings.weight_decay,
                settings.max_grad_norm,
                config.dataset.seed,
                device,
                run,
                share,
            )
        path = heads_path(tracked)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(heads, path)
        log_checkpoint(run, path, tracked, HEADS)
        run.finish()
    # Score the logged heads on the test tiles and draw the qualitative ones
    if evaluate or not finetune:
        run, mae = tracked_run(BENCHMARK)
        heads = torch.load(logged_checkpoint(run, tracked, HEADS), map_location="cpu")
        evaluation.evaluate(
            {
                share: built_benchmark_model(config, mae, device, heads[share])
                for share in shares
            },
            build,
            test,
            targets,
            config.benchmark.instruments,
            sizes,
            shapes,
            config.benchmark.batch_size,
            config.benchmark.evaluation.tiles,
            config.benchmark.evaluation.shown,
            config.dataset.seed,
            workers,
            device,
            run,
        )
        run.finish()


if __name__ == "__main__":
    run_stage(
        BENCHMARK,
        run_benchmark,
        __doc__,
        {
            "finetune": "fine-tune and log the heads alone",
            "evaluate": "evaluate the logged heads alone",
        },
    )
