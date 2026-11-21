"""Where the repository keeps what a run reads and writes, and what files are named."""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

ENV_PATH = REPO_ROOT / ".env"

CONFIGS_ROOT = REPO_ROOT / "configs"

ARTIFACTS_ROOT = REPO_ROOT / "artifacts"

RESULTS_ROOT = REPO_ROOT / "results"

RESULTS_FILE = "results.parquet"

RECONSTRUCTION_FILE = "reconstruction-{}-{}.npz"

RANGING_FILE = "ranging-{}-{}.npz"


def build_root(build: str, root: str) -> Path:
    """Return where one build of the dataset sits on this machine.

    Args:
        build: The build's name.
        root: The folder holding the builds, relative to the repository.

    Returns:
        root: The build's folder.
    """
    return REPO_ROOT / root / build


def ready_array_path(
    identity: str, name: str, setup: Sequence[object], key: str
) -> str:
    """Return where one array of one tile's patches is kept, under its build's root.

    Args:
        identity: The tile.
        name: The instrument, as ODE names it.
        setup: What the patches were cut under, hashed into the folder name.
        key: Which of the patch arrays.

    Returns:
        path: The array's file, relative to the build's root.
    """
    return f"ready/{identity}/{name}-{zlib.crc32(repr(setup).encode()):08x}/{key}.npy"


def checkpoint_path(checkpoints: str, name: str) -> Path:
    """Return where one checkpoint is written on this machine.

    Args:
        checkpoints: The folder holding the checkpoints, relative to the repository.
        name: What the checkpoint is called.

    Returns:
        path: The checkpoint's file.
    """
    return REPO_ROOT / checkpoints / f"{name}.pt"


def results_directory(run_name: str, part: str) -> Path:
    """Return where one part of a run's evaluation is written on this machine.

    Args:
        run_name: What the run is called.
        part: The part, such as "reconstruction".

    Returns:
        directory: The part's folder.
    """
    return RESULTS_ROOT / run_name / part
