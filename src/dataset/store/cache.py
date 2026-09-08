"""What a read keeps of the crops it fetched, until the room it was given runs out."""

from __future__ import annotations

import os
import tempfile
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Cache:
    """The crops a read has fetched, kept on the machine it is running on.

    Attributes:
        root: The directory they are kept in, made when the first one lands.
        budget: How many bytes may be kept before the least recently read go.
        held: How large each kept crop is, in the order they were last read.
        held_bytes: How much the kept crops come to.
    """

    root: Path
    budget: int
    held: OrderedDict[str, int] = field(default_factory=OrderedDict)
    held_bytes: int = 0

    def __post_init__(self) -> None:
        """Take up whatever an earlier run of the same job left in the directory."""
        found = sorted(
            (one for one in self.root.rglob("*") if one.is_file()),
            key=lambda one: one.stat().st_mtime,
        )
        for one in found:
            self.held[str(one.relative_to(self.root))] = one.stat().st_size
        self.held_bytes = sum(self.held.values())

    def kept(self, path: str) -> bytes | None:
        """Return one crop the cache already holds, and None where it holds none.

        Args:
            path: Where the crop sits, relative to the build's own root.

        Returns:
            data: What the crop holds, or None where it was never fetched or has
                since been dropped to make room.
        """
        if path not in self.held:
            return None
        self.held.move_to_end(path)
        try:
            return (self.root / path).read_bytes()
        except FileNotFoundError:
            self.held_bytes -= self.held.pop(path)
            return None

    def keep(self, path: str, data: bytes) -> None:
        """Keep one fetched crop, dropping the least recently read to make room.

        Args:
            path: Where the crop sits, relative to the build's own root.
            data: What it holds.
        """
        if len(data) > self.budget:
            return
        while self.held and self.held_bytes + len(data) > self.budget:
            dropped, size = self.held.popitem(last=False)
            (self.root / dropped).unlink(missing_ok=True)
            self.held_bytes -= size
        held = self.root / path
        held.parent.mkdir(parents=True, exist_ok=True)
        handle, staged = tempfile.mkstemp(dir=held.parent)
        with os.fdopen(handle, "wb") as writing:
            writing.write(data)
        os.replace(staged, held)
        self.held[path] = len(data)
        self.held_bytes += len(data)
