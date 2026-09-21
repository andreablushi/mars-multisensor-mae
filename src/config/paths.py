"""Where the repository keeps what a run reads, and what a build lands as."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIGS_ROOT = REPO_ROOT / "configs"


def build_root(build: str, root: str) -> Path:
    """Return where one build of the dataset sits on this machine.

    Args:
        build: The build's own name, which is the directory it owns.
        root: Where every build sits, relative to the repository.

    Returns:
        path: That build's own directory, which need not exist yet.
    """
    return REPO_ROOT / root / build
