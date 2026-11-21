"""What a run reads from DigitalHub: the published builds."""

from __future__ import annotations

from dataclasses import replace
from urllib.parse import urlparse

import digitalhub as dh
from botocore.exceptions import ClientError, ResponseStreamingError
from urllib3.exceptions import SSLError

from mars_multisensor_mae.configs.paths import build_root
from mars_multisensor_mae.configs.schema import Config
from mars_multisensor_mae.dataset.store import DatasetBuild

BROKEN_READS = (ClientError, ResponseStreamingError, SSLError)


def published_dataset_build(config: Config, build: str) -> DatasetBuild:
    """Return one published build of the dataset, read from the platform's store.

    Args:
        config: The run's config.
        build: The build to read, published as <version>-dataset-<build>.

    Returns:
        build: The build, off disk where already fetched and from the store otherwise.
    """
    return published_build(config, f"{config.dataset.version}-dataset-{build}")


def published_build(config: Config, name: str, subfolder: str = "") -> DatasetBuild:
    """Return one published build, read from a folder of the platform's store.

    Args:
        config: The run's config.
        name: The artifact the build is published in.
        subfolder: The folder of the artifact the build sits in, or "" for its top.

    Returns:
        build: The build, off disk where already fetched and from the store otherwise.
    """
    project = dh.get_or_create_project(config.digitalhub.project)
    published = urlparse(project.get_artifact(name).spec.path)
    bucket = published.netloc
    held_in = f"{subfolder}/" if subfolder else ""
    prefix = published.path.strip("/") + "/" + held_in
    client = dh.get_s3_client()

    def fetch(path: str) -> bytes:
        """Return what one object of the build holds, straight from the store.

        Args:
            path: The object, relative to the build.

        Returns:
            data: The object's bytes.

        Raises:
            RuntimeError: When the read breaks again once the credentials are minted.
        """
        nonlocal client
        key = prefix + path
        try:
            return client.get_object(Bucket=bucket, Key=key)["Body"].read()
        except BROKEN_READS:
            # The store's credentials lapse and its streams break mid run.
            dh.refresh_token()
            client = dh.get_s3_client()
            try:
                return client.get_object(Bucket=bucket, Key=key)["Body"].read()
            except BROKEN_READS as refused:
                raise RuntimeError(f"{key}: {refused}") from None

    held = DatasetBuild(
        root=build_root(name, config.dataset.root) / subfolder,
        fetch=fetch,
        clips=config.dataset.clip,
        sharad_atmosphere_cropping=config.dataset.sharad_atmosphere_cropping,
        acquisition={
            name: tuple(fields) if config.training.acquisition else ()
            for name, fields in config.dataset.acquisition.items()
        },
    )
    if config.dataset.stats_version != config.dataset.version:
        trained = replace(
            config,
            dataset=replace(config.dataset, version=config.dataset.stats_version),
        )
        held.stats = published_dataset_build(trained, config.dataset.build).read_stats()
    return held
