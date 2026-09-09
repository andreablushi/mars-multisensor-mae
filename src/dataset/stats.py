"""What each instrument's values run to over a whole build, pooled from its index."""

from __future__ import annotations

import math
from collections.abc import Iterable

from building.metadata.observation import ObservationMetadata


def pooled_statistics(
    observations: Iterable[ObservationMetadata],
) -> dict[str, dict[str, float]]:
    """Return what each instrument's values run to, without reading one crop.

    Args:
        observations: The index rows to pool, which carry the mean and the
            deviation each crop was measured to hold.

    Returns:
        statistics: One entry per instrument that measured anything, keyed as
            ODE names it, holding how many values it was pooled from, their
            mean, and their deviation worked from the moments the rows carry.
    """
    standing: dict[str, list[ObservationMetadata]] = {}
    for one in observations:
        if one.valid_count and _finite(one.value_mean) and _finite(one.value_std):
            standing.setdefault(one.instrument, []).append(one)
    statistics = {}
    for instrument, held in standing.items():
        total = sum(one.valid_count for one in held)
        mean = sum(one.valid_count * one.value_mean for one in held) / total
        second = (
            sum(
                one.valid_count * (one.value_std**2 + one.value_mean**2) for one in held
            )
            / total
        )
        statistics[instrument] = {
            "count": total,
            "mean": mean,
            "deviation": math.sqrt(max(second - mean**2, 0.0)),
        }
    return statistics


def _finite(held: float | None) -> bool:
    """Say whether one statistic the index carries is a number to pool at all.

    Args:
        held: The statistic, which a crop measuring nothing leaves unset and a
            crop of a sounder can carry as not a number.

    Returns:
        finite: Whether it is a number the pooling can stand on.
    """
    return held is not None and math.isfinite(held)
