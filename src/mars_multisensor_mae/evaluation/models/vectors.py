"""One pooling's vector of every sample for each instrument and all of them."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mars_multisensor_mae.architecture.models.context import Pooling

ALL_VIEW = "all"


@dataclass(frozen=True, slots=True)
class TileVectors:
    """One pooling's vector of every sample for each instrument, one row per sample.

    Attributes:
        pooling: How the tokens were pooled.
        identities: The samples, tiles or crops, in row order. (T,)
        instruments: Each instrument's vectors. (T, D)
        fused: The vectors of all the instruments together, the model's CLS or the
            mean of their GAP, or None when none was made. (T, D)
    """

    pooling: Pooling
    identities: list[str]
    instruments: dict[str, np.ndarray]
    fused: np.ndarray | None

    def views(self) -> dict[str, np.ndarray]:
        """Return every view's vectors: each instrument, then "all" when there is one.

        Returns:
            views: Each view's vectors, keyed by instrument or ALL_VIEW. (T, D)
        """
        if self.fused is None:
            return dict(self.instruments)
        return {**self.instruments, ALL_VIEW: self.fused}

    def sorted_by_identity(self) -> TileVectors:
        """Return the same vectors, their rows in the order of their samples' names.

        Returns:
            vectors: The same pooling, rows sorted by identity.
        """
        order = np.argsort(self.identities, kind="stable")
        return TileVectors(
            self.pooling,
            [self.identities[at] for at in order],
            {name: held[order] for name, held in self.instruments.items()},
            None if self.fused is None else self.fused[order],
        )
