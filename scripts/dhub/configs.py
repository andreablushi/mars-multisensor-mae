"""Reading `configs/digitalhub.yaml`, the one file a platform run is settled from."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from configs.paths import PLATFORM_CONFIG_PATH


@dataclass(frozen=True, slots=True)
class Platform:
    """What a run submitted to DigitalHub is given.

    Attributes:
        project: The project every run and every published model belongs to.
        repository: The repository the platform clones when a job starts.
        python_version: The interpreter a job runs on.
        base_image: The platform's own base image a job runs on.
        stages: What each stage is registered as and gets: function, profile and
            resources.
        volume: The volume a run keeps its ready tiles on, as the platform takes it.
    """

    project: str
    repository: str
    python_version: str
    base_image: str
    stages: dict[str, dict]
    volume: dict


@lru_cache(maxsize=1)
def load_platform(path: Path = PLATFORM_CONFIG_PATH) -> Platform:
    """Return what a platform run is given, reading the config file once.

    Args:
        path: The config file, which carries every setting a run is submitted with.

    Returns:
        platform: The settled choices for the submission.
    """
    return Platform(**yaml.safe_load(path.read_text(encoding="utf-8")))


def stage_workers(stage: str) -> int:
    """Return how many processes read tiles beside one stage's own work.

    Args:
        stage: Which stage, whose cores are what a box running it holds.

    Returns:
        workers: Those cores, which a run here reads as a run on the platform does.
    """
    return int(load_platform().stages[stage]["resources"]["cpu"])
