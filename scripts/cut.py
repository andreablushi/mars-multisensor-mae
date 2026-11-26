"""Cutting every tile a config reads into its cut tiles, for stages that never cut."""

from __future__ import annotations

import argparse
import gc
import os
from functools import partial

from mars_multisensor_dataset.building.configs import sharad
from torch.utils.data import DataLoader
from utils.build import DISK, published_dataset_build, published_lavaflow_build

from mars_multisensor_mae.configs.load import load_config
from mars_multisensor_mae.configs.paths import READY_MARKER
from mars_multisensor_mae.dataset.loader import cut_missing_tiles, ready_setup_folders
from mars_multisensor_mae.dataset.models.patch import PatchLayout
from mars_multisensor_mae.dataset.store import DatasetBuild
from mars_multisensor_mae.evaluation.lavaflow import lavaflow_crop_build
from mars_multisensor_mae.logs.tracker import console_logger

log = console_logger(__name__)

CUT_BATCH = 16

LOGGED_BATCHES = 100


def cut_every_tile(build: DatasetBuild, layout: PatchLayout, workers: int) -> None:
    """Cut every tile of one build missing from its cut tiles, then mark them complete.

    Args:
        build: The build the tiles are read from.
        layout: How each instrument is cut into patches, and the shape the model
            reads one in.
        workers: How many processes cut tiles.
    """
    tiles = sorted(build.read_observation_metadata())
    gc.freeze()
    loader = DataLoader(
        tiles,
        batch_size=CUT_BATCH,
        num_workers=workers,
        collate_fn=partial(cut_missing_tiles, build=build, layout=layout),
    )
    cut = 0
    for batch, done in enumerate(loader, 1):
        cut += done
        if batch % LOGGED_BATCHES == 0:
            log.info(
                "%s: %d of %d read, %d cut",
                build.root,
                batch * CUT_BATCH,
                len(tiles),
                cut,
            )
    for folder in ready_setup_folders(build, layout).values():
        (folder / READY_MARKER).touch()
    log.info("%s: %d tiles complete, %d cut now", build.root, len(tiles), cut)


def run_cut(overrides: list[str]) -> None:
    """Cut every tile of the builds a config's stages read, each build marked once done.

    Args:
        overrides: What to change in the configs, as dotted keys, as a stage is given.

    Raises:
        ValueError: When the builds would be fetched from the platform.
    """
    config = load_config(overrides)
    if config.dataset.store != DISK:
        raise ValueError(f"the cut reads its builds off disk: set dataset.store={DISK}")
    training = published_dataset_build(config, config.dataset.build)
    evaluation = published_dataset_build(config, config.evaluation.build)
    sizes = {name: config.dataset.patchsize[name] for name in config.model.instruments}
    layout = PatchLayout.from_index(evaluation.read_observation_metadata(), sizes)
    cuts = [
        (training, PatchLayout.from_index(training.read_observation_metadata(), sizes)),
        (evaluation, layout),
    ]
    name = sharad.LAYOUT.instrument
    if (
        config.evaluation.lavaflow.build is not None
        and name in config.model.instruments
    ):
        lavaflow = published_lavaflow_build(config)
        crops = lavaflow_crop_build(lavaflow, lavaflow.read_lavaflow_crops())
        cuts.append((crops, layout.only([name])))
    workers = len(os.sched_getaffinity(0))
    for build, cut in cuts:
        cut_every_tile(build, cut, workers)


if __name__ == "__main__":
    parsed = argparse.ArgumentParser(description=__doc__)
    parsed.add_argument(
        "overrides", nargs="*", help="what to change in the configs, as dotted keys"
    )
    run_cut(parsed.parse_args().overrides)
