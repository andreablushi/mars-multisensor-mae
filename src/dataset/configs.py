"""Reading the config beside this module, the one place a read is settled from."""

from __future__ import annotations

from pathlib import Path

import yaml

from dataset.models.settings import Settings

CONFIG_PATH = Path(__file__).parent / "config.yaml"

GIB = 1024**3


def load(
    path: Path = CONFIG_PATH,
    build: str | None = None,
    per_instrument: int | None = None,
) -> Settings:
    """Settle what a read should do, reading the config file once.

    Args:
        path: The config file, which carries every choice a read is made with.
        build: The build to read, standing in for the config where a run names
            one of its own.
        per_instrument: How many observations of each instrument a sample draws,
            standing in for the config the same way.

    Returns:
        choices: The settled choices for the read.
    """
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Settings(
        project=config["project"],
        build=build or config["build"],
        per_instrument=per_instrument or config["per_instrument"],
        per_observation=config["per_observation"],
        tiles=config["tiles"],
        keep_valid=config["keep_valid"],
        cache_bytes=int(config["cache_gb"] * GIB),
        shares=config["split"],
        seed=config["seed"],
    )


def cache_root(path: Path = CONFIG_PATH) -> Path:
    """Return where a run keeps the crops it has already fetched.

    Args:
        path: The config file, which names the directory.

    Returns:
        path: The directory, which need not exist.
    """
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Path(config["cache_dir"]).expanduser()
