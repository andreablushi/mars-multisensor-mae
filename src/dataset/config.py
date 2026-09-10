"""The config beside this module, the one place a read is settled from."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Return every choice a read is made with.

    Args:
        path: The config file, which carries all of them.

    Returns:
        config: The choices, as the file spells them.
    """
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_root(config: dict) -> Path:
    """Return where the build a read is made of sits on this machine.

    Args:
        config: The choices a read is made with, which name the build and the
            directory every build sits under.

    Returns:
        root: That build's own directory, which need not exist yet.
    """
    return REPO_ROOT / config["root"] / config["build"]
