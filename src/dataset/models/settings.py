"""The settled choices for a read, and for how it cuts what it reads."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT = "default"


@dataclass(frozen=True, slots=True)
class Settings:
    """The settled choices for a read, taken from one config file beside the code.

    Attributes:
        project: The DigitalHub project the build is published in.
        build: The build to read, which is published as "dataset-<build>".
        per_instrument: How many observations of each instrument a sample draws.
        per_observation: How many patches a sample draws of each of them. One
            crop holds far more than a pass over it wants: a CTX scan runs to
            millions of them.
        tiles: How far a patch runs along each kind of axis, keyed first by the
            instrument. An axis whose kind is not named is kept whole, and an
            instrument that is not named takes the default entry.
        keep_valid: The share of a patch that must be measured for it to be kept.
        cache_bytes: How much of the fetched crops the run keeps on its own disk.
        shares: What share of the features each split holds, keyed by its name.
        seed: The number that fixes every draw and the split.
    """

    project: str
    build: str
    per_instrument: int
    per_observation: int
    tiles: dict[str, dict[str, int]]
    keep_valid: float
    cache_bytes: int
    shares: dict[str, float]
    seed: int

    def tiles_of(self, instrument: str) -> dict[str, int]:
        """Return how far a patch of one instrument runs along each kind of axis.

        Args:
            instrument: The instrument that took it, as ODE names it.

        Returns:
            tiles: What that instrument is cut by, and what every instrument is
                cut by where it names none of its own.
        """
        return self.tiles.get(instrument, self.tiles[DEFAULT])

    @property
    def artifact(self) -> str:
        """Return the name the build is published under.

        Returns:
            name: The artifact name, which is the build's own name prefixed.
        """
        return f"dataset-{self.build}"
