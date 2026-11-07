"""Benchmarking one logged model on SWIM's ice: run here by default, or with --dh."""

from __future__ import annotations

from dataclasses import asdict

from dhub.store import published_dataset_build, published_ice_labels
from dhub.submit import requested_read_workers, run_stage
from digitalhub_runtime_python import handler
from evaluate import trained_model

from benchmarking.benchmark import benchmark
from configs.load import load_config
from configs.model import built_model
from logs.benchmark import BENCHMARK
from logs.tracker import console_logger, start_logging

log = console_logger(__name__)


@handler()
def run_benchmark(run_name: str, overrides: list[str] | None = None) -> None:
    """Probe and fine-tune heads over one logged model on SWIM's ice, and log them.

    Args:
        run_name: What the run is called, which names the model read and the
            tracked run, a last part of "random" benchmarking the seed's own weights.
        overrides: What to change in the configs, as dotted keys.
    """
    config = load_config(overrides or [])
    build = published_dataset_build(config, config.benchmark.build)
    model, device, sizes, shapes = built_model(config, build)
    workers = requested_read_workers(BENCHMARK)
    left = [
        one
        for one in config.model.instruments
        if one not in config.benchmark.instruments
    ]
    tracked = run_name + "".join(f"-no-{one.lower()}" for one in left)
    run = start_logging(asdict(config), tracked, BENCHMARK, {"device": str(device)})
    log.info("benchmarking %s on %s", tracked, device)
    benchmark(
        trained_model(model, run, run_name),
        build,
        published_ice_labels(),
        config.benchmark.instruments,
        sizes,
        shapes,
        config.model.encoder_dim,
        config.model.encoder_heads,
        config.benchmark.segmenter_depth,
        config.benchmark.folds,
        config.benchmark.block_deg,
        config.benchmark.pooling,
        config.benchmark.finetune.budget,
        config.benchmark.finetune.budget_scale,
        config.benchmark.finetune.batch_size,
        config.benchmark.evaluation.batch_size,
        config.benchmark.finetune.epochs,
        config.benchmark.finetune.patience,
        config.benchmark.finetune.warmup_epochs,
        config.benchmark.finetune.head_learning_rate,
        config.benchmark.finetune.encoder_learning_rate,
        config.benchmark.finetune.weight_decay,
        config.benchmark.finetune.max_grad_norm,
        config.benchmark.evaluation.threshold,
        config.benchmark.evaluation.tiles,
        config.benchmark.evaluation.shown,
        config.dataset.seed,
        workers,
        device,
        run,
        tracked,
    )
    run.finish()


if __name__ == "__main__":
    run_stage(BENCHMARK, run_benchmark, __doc__)
