"""Where the repository keeps what a run reads and writes, and what files are named."""

from __future__ import annotations

import os
import zlib
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

ENV_PATH = REPO_ROOT / ".env"

CONFIGS_ROOT = REPO_ROOT / "configs"

OUTPUT_ROOT = Path(os.environ.get("MAE_OUTPUT_ROOT", REPO_ROOT))

ARTIFACTS_ROOT = OUTPUT_ROOT / "artifacts"

RESULTS_ROOT = OUTPUT_ROOT / "results"

RESULTS_FILE = "results.parquet"

RECONSTRUCTION_FILE = "reconstruction-{}-{}.npz"

RANGING_FILE = "ranging-{}-{}.npz"

READY_ROOT_NAME = "ready"

READY_TILE_FILE = "{}/{}.npz"

READY_MARKER = "complete"


def build_root(build: str, root: str) -> Path:
    """Return where one build of the dataset sits on this machine.

    Args:
        build: The build's name.
        root: The folder holding the builds, relative to the repository or absolute.

    Returns:
        root: The build's folder.
    """
    return REPO_ROOT / root / build


def ready_setup_folder(name: str, setup: Sequence[object]) -> str:
    """Return the folder one instrument's tiles cut under one setup are kept in.

    Args:
        name: The instrument, as ODE names it.
        setup: What the patches were cut under, hashed into the folder name.

    Returns:
        folder: The folder, relative to the build's cut tiles.
    """
    return f"{name}-{zlib.crc32(repr(setup).encode()):08x}"


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
