"""The builds a run reads, off the platform's store or off disk alone."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import NoReturn

from mars_multisensor_mae.configs.paths import build_root
from mars_multisensor_mae.configs.schema import Config
from mars_multisensor_mae.dataset.store import DatasetBuild

DIGITALHUB = "digitalhub"

DISK = "disk"


def missing_object(root: Path, path: str) -> NoReturn:
    """Fail on an object a build read off disk alone does not hold.

    Args:
        root: Where the build sits.
        path: The object, relative to the build.

    Raises:
        FileNotFoundError: Always, naming the object.
    """
    raise FileNotFoundError(f"{root / path} is missing, and dataset.store={DISK}")


def published_dataset_build(config: Config, build: str) -> DatasetBuild:
    """Return one published build of the dataset, read as dataset.store says.

    Args:
        config: The run's config.
        build: The build to read, published as <version>-dataset-<build>.

    Returns:
        build: The build, off disk where held and from the store otherwise.
    """
    return published_build(config, f"{config.dataset.version}-dataset-{build}")


def published_lavaflow_build(config: Config) -> DatasetBuild:
    """Return the lava flow build the retrieval reads, read as dataset.store says.

    Args:
        config: The run's config, naming the build in evaluation.lavaflow.build.

    Returns:
        build: The SHARAD crops and their verdicts.
    """
    named = f"{config.dataset.version}-{config.evaluation.lavaflow.build}"
    return published_build(config, named, "dataset")


def published_build(config: Config, name: str, subfolder: str = "") -> DatasetBuild:
    """Return one published build, read from a folder of it as dataset.store says.

    Args:
        config: The run's config.
        name: The artifact the build is published in.
        subfolder: The folder of the artifact the build sits in, or "" for its top.

    Returns:
        build: The build, off disk where held and from the store otherwise, which
            cuts its missing tiles only when read from the store.

    Raises:
        ValueError: When dataset.store names neither the platform nor the disk.
    """
    root = build_root(name, config.dataset.root) / subfolder
    store = config.dataset.store
    if store == DISK:
        fetch = partial(missing_object, root)
    elif store == DIGITALHUB:
        from dhub.store import published_fetch

        fetch = published_fetch(config, name, subfolder)
    else:
        raise ValueError(f"dataset.store={store} is not {DIGITALHUB} or {DISK}")
    ready = config.dataset.ready_root
    held = DatasetBuild(
        root=root,
        fetch=fetch,
        clips=config.dataset.clip,
        sharad_atmosphere_cropping=config.dataset.sharad_atmosphere_cropping,
        acquisition={
            name: tuple(fields) if config.training.acquisition else ()
            for name, fields in config.dataset.acquisition.items()
        },
        ready_root=None if ready is None else build_root(name, ready) / subfolder,
        cuts=store == DIGITALHUB,
    )
    if config.dataset.stats_version != config.dataset.version:
        trained = replace(
            config,
            dataset=replace(config.dataset, version=config.dataset.stats_version),
        )
        held.stats = published_dataset_build(trained, config.dataset.build).read_stats()
    return held
