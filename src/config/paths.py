"""Where the repository keeps what a run reads, and what a build lands as."""

from __future__ import annotations

from pathlib import Path

from config.schema import DatasetConfig

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIGS_ROOT = REPO_ROOT / "configs"


def build_root(dataset: DatasetConfig) -> Path:
    """Return where the build a read is made of sits on this machine.

    Args:
        dataset: What a run reads, which names the build and the directory every
            build sits under.

    Returns:
        root: That build's own directory, which need not exist yet.
    """
    return REPO_ROOT / dataset.root / dataset.build
