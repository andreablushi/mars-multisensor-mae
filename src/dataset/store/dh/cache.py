"""What a read keeps of the crops it fetched, so a later pass fetches none of them."""

from __future__ import annotations

from pathlib import Path

from shared.disk.files import atomic_path


def kept(root: Path, path: str) -> bytes | None:
    """Return one crop the cache already holds, and None where it holds none.

    Args:
        root: Where the run keeps what it has fetched.
        path: Where the crop sits, relative to the build's own root.

    Returns:
        data: What the crop holds, or None where it was never fetched.
    """
    held = root / path
    return held.read_bytes() if held.is_file() else None


def keep(root: Path, path: str, data: bytes) -> None:
    """Keep one fetched crop, written whole or not at all.

    Args:
        root: Where the run keeps what it has fetched.
        path: Where the crop sits, relative to the build's own root.
        data: What it holds.
    """
    with atomic_path(root / path) as staged:
        staged.write_bytes(data)
