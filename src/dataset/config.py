"""The config beside this module, the one place a read is settled from."""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Return every choice a read is made with.

    Args:
        path: The config file, which carries all of them.

    Returns:
        config: The choices, as the file spells them.
    """
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def patchsize_of(config: dict, instrument: str) -> int:
    """Return how far a patch of one instrument runs along every axis it is cut on.

    Args:
        config: The choices a read is made with.
        instrument: The instrument that took it, as ODE names it.

    Returns:
        patchsize: What that instrument is cut by, or what everything unnamed is.
    """
    return config["patchsize"].get(instrument, config["patchsize"]["default"])
