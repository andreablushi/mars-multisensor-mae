"""Embedding the labelled tiles with every published model and a random one."""

from __future__ import annotations

import numpy as np
import torch
from dhub.store import publish_results, published_build, published_checkpoint
from dhub.submit import run_stage, stage_workers
from digitalhub_runtime_python import handler
from evaluate import EVALUATION_STAGE, built_model

from configs.load import load_config
from configs.paths import RESULTS_ROOT
from dataset.loader import tile_loader
from evaluation.evaluate import TileTokens
from evaluation.metrics import chamfer_distances
from training.checkpoint import load_checkpoint
from training.step import device_batch

RUNS = ("mae", "mae-clean-crism", None)
BENCHMARK_HANDLER = "scripts.benchmark:run_benchmark"


@handler()
def run_benchmark(project=None, overrides: list[str] | None = None) -> None:
    """Keep every tile's pooled tokens and Chamfer distances, per model and instrument.

    Args:
        project: The DigitalHub project the results are published in, or None here.
        overrides: What to compose the config with, as hydra spells them.
    """
    config = load_config(overrides or [])
    build = published_build(config.evaluation.build, config.dataset.root)
    classes = build.read_label_by_tile()
    by_tile = build.read_observation_metadata()
    tiles = sorted(classes)
    kept = {"tiles": np.array(tiles), "labels": np.array([classes[t] for t in tiles])}
    for run in RUNS:
        torch.manual_seed(config.dataset.seed)
        model, device, sizes, shapes = built_model(config, build)
        name = run or "random"
        if run is not None:
            checkpoint = published_checkpoint(run, config.training.checkpoints)
            load_checkpoint(checkpoint, model)
        model.eval()
        loader = tile_loader(
            build,
            {tile: by_tile[tile] for tile in tiles},
            build.read_axes_by_instrument(),
            sizes,
            config.dataset.pool,
            shapes,
            config.model.delay,
            config.training.memory_batch_size,
            stage_workers(EVALUATION_STAGE),
            shuffle=False,
            budget=None,
            seed=None,
        )
        generator = torch.Generator().manual_seed(config.dataset.seed)
        held = {variant: {} for variant in ("normal", "shuffled")}
        with torch.no_grad():
            for batch, identities in loader:
                batch = device_batch(batch, device)
                present = {one: tokens.present for one, tokens in batch.items()}
                for variant in held:
                    if variant == "shuffled":
                        # Each tile's positions permuted among its own patches
                        for one, tokens in batch.items():
                            for at in range(len(identities)):
                                slots = present[one][at].nonzero().squeeze(1)
                                order = torch.randperm(len(slots), generator=generator)
                                tokens.position[at, slots] = tokens.position[
                                    at, slots[order.to(device)]
                                ]
                    with torch.autocast(device.type, dtype=torch.bfloat16):
                        embedded = model.embed(batch, present)
                    for one, values in embedded.items():
                        for at, identity in enumerate(identities):
                            mask = present[one][at]
                            if not mask.any():
                                continue
                            held[variant].setdefault(one, {})[identity] = TileTokens(
                                values[at, mask].float(),
                                batch[one].position[at, mask, :2],
                            )
                            if run is None and variant == "normal":
                                # The mean patch, a baseline that reads no model
                                raw = batch[one].values[at, mask].flatten(1).mean(0)
                                kept.setdefault(f"raw/{one}", {})[identity] = (
                                    raw.cpu().numpy()
                                )
        for variant, by_instrument in held.items():
            for one, tokens in by_instrument.items():
                # A tile without this instrument stays, as NaN
                holding = [at for at, tile in enumerate(tiles) if tile in tokens]
                ordered = [tokens[tiles[at]] for at in holding]
                pooled = np.full((len(tiles), ordered[0].values.shape[-1]), np.nan)
                pooled[holding] = np.stack(
                    [
                        torch.nn.functional.normalize(t.values.mean(0), dim=0)
                        .cpu()
                        .numpy()
                        for t in ordered
                    ]
                )
                chamfer = np.full((len(tiles), len(tiles)), np.nan)
                chamfer[np.ix_(holding, holding)] = (
                    chamfer_distances(ordered).double().cpu().numpy()
                )
                kept[f"{name}/{variant}/{one}/pooled"] = pooled
                kept[f"{name}/{variant}/{one}/chamfer"] = chamfer
                print(name, variant, one, len(holding), "tiles", flush=True)
    for key in [k for k in kept if k.startswith("raw/")]:
        width = len(next(iter(kept[key].values())))
        kept[key] = np.stack(
            [kept[key].get(tile, np.full(width, np.nan)) for tile in tiles]
        )
    path = RESULTS_ROOT / "benchmark" / "benchmark.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **kept)
    print("written", path, flush=True)
    if project is not None:
        publish_results(project, path, "benchmark")


if __name__ == "__main__":
    run_stage(EVALUATION_STAGE, BENCHMARK_HANDLER, run_benchmark, __doc__)
