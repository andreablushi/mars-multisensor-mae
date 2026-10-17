"""The strategies an instrument's values are normalised by, from build constants."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

STRATEGIES = ("zscore", "zscore-global", "minmax", "minmax-global")


def offset_and_spread(
    constants: Mapping[str, Any], strategy: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return what one strategy subtracts from an instrument's values and divides by.

    Args:
        constants: The instrument's mean, std, min and max, per band where it has
            bands, with the ones pooled over every band under `global`.
        strategy: One of STRATEGIES, -global sharing one value across the bands.

    Returns:
        offset: The mean or the min, per band or one.
        spread: The std or max - min, per band or one.

    Raises:
        ValueError: When the strategy is none of STRATEGIES.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"{strategy} is none of {STRATEGIES}.")
    kind, _, pooled = strategy.partition("-")
    if pooled:
        # An instrument without bands holds one value already
        constants = constants.get("global", constants)
    if kind == "zscore":
        mean, std = (
            np.asarray(constants[name], np.float32) for name in ("mean", "std")
        )
        return mean, std
    low, high = (np.asarray(constants[name], np.float32) for name in ("min", "max"))
    return low, high - low
