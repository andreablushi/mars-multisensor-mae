"""Where the repository keeps what a run reads and writes, and what files are named."""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ENV_PATH = REPO_ROOT / ".env"

CONFIGS_ROOT = REPO_ROOT / "configs"

PLATFORM_CONFIG_PATH = CONFIGS_ROOT / "digitalhub.yaml"

ARTIFACTS_ROOT = REPO_ROOT / "artifacts"

RESULTS_ROOT = REPO_ROOT / "results"

RESULTS_FILE = "results.parquet"

RECONSTRUCTION_FILE = "reconstruction-{}.npz"

ATTENTION_FILE = "attention-{}-{}.npz"

RANGING_FILE = "ranging-{}-{}.npz"

TOKENS_FILE = "tokens-{}.npz"


def build_root(build: str, root: str) -> Path:
    """Return where one build of the dataset sits on this machine.

    Args:
        build: The build's own name, which is the directory it owns.
        root: Where every build sits, relative to the repository.

    Returns:
        path: That build's own directory, which need not exist yet.
    """
    return REPO_ROOT / root / build


def ready_array_path(
    identity: str, name: str, setup: Sequence[object], key: str
) -> str:
    """Return where one array of one tile's patches is kept, under its build's root.

    Args:
        identity: The tile.
        name: The instrument.
        setup: What the patches were cut with, so another setup keeps its own.
        key: Which of its patch arrays.

    Returns:
        path: That array's file, relative to the build root.
    """
    return f"ready/{identity}/{name}-{zlib.crc32(repr(setup).encode()):08x}/{key}.npy"


def checkpoint_path(checkpoints: str, name: str) -> Path:
    """Return where one checkpoint is written on this machine.

    Args:
        checkpoints: Where checkpoints land, relative to the repository.
        name: What the checkpoint is called.

    Returns:
        path: That checkpoint's file.
    """
    return REPO_ROOT / checkpoints / f"{name}.pt"


def results_path(run_name: str, name: str = RESULTS_FILE) -> Path:
    """Return where one of a run's evaluation files is written on this machine.

    Args:
        run_name: What the run is called.
        name: Which file, its results by default.

    Returns:
        path: That run's file.
    """
    return RESULTS_ROOT / run_name / name
