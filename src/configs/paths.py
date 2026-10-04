"""Where the repository keeps what a run reads and writes, and what files are named."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ENV_PATH = REPO_ROOT / ".env"

CONFIGS_ROOT = REPO_ROOT / "configs"

PLATFORM_CONFIG_PATH = CONFIGS_ROOT / "digitalhub.yaml"

RESULTS_ROOT = REPO_ROOT / "results"

RESULTS_FILE = "results.parquet"


def build_root(build: str, root: str) -> Path:
    """Return where one build of the dataset sits on this machine.

    Args:
        build: The build's own name, which is the directory it owns.
        root: Where every build sits, relative to the repository.

    Returns:
        path: That build's own directory, which need not exist yet.
    """
    return REPO_ROOT / root / build


def ready_tile_path(identity: str) -> str:
    """Return where one tile's packed patches are kept, under its build's root.

    Args:
        identity: The tile.

    Returns:
        path: That file, relative to the build root.
    """
    return f"ready/{identity}.npz"


def checkpoint_path(checkpoints: str, name: str) -> Path:
    """Return where one checkpoint is written on this machine.

    Args:
        checkpoints: Where checkpoints land, relative to the repository.
        name: What the checkpoint is called.

    Returns:
        path: That checkpoint's file.
    """
    return REPO_ROOT / checkpoints / f"{name}.pt"


def results_path(run_name: str) -> Path:
    """Return where one run's evaluation results are written on this machine.

    Args:
        run_name: What the run is called.

    Returns:
        path: That run's results file.
    """
    return RESULTS_ROOT / run_name / RESULTS_FILE
