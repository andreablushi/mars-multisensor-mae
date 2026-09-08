"""What each instrument's values run to over a whole build, pooled from its index."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
from building.metadata.observation import ObservationMetadata

from dataset.models.statistics import Statistics


def pooled_statistics(
    observations: Iterable[ObservationMetadata],
) -> dict[str, Statistics]:
    """Return what each instrument's values run to, without reading one crop.

    Args:
        observations: The index rows to pool, which carry the mean and the
            deviation each crop was measured to hold.

    Returns:
        statistics: One entry per instrument that measured anything, keyed as
            ODE names it. The per band entries are set only for an instrument
            whose rows carry them.
    """
    standing: dict[str, list[ObservationMetadata]] = {}
    for one in observations:
        if one.valid_count and _finite(one.value_mean) and _finite(one.value_std):
            standing.setdefault(one.instrument, []).append(one)
    statistics = {}
    for instrument, held in standing.items():
        mean, deviation = _pooled(
            [one.valid_count for one in held],
            [one.value_mean for one in held],
            [one.value_std for one in held],
        )
        band_mean, band_deviation = _banded(held)
        statistics[instrument] = Statistics(
            instrument=instrument,
            count=sum(one.valid_count for one in held),
            mean=mean,
            deviation=deviation,
            band_mean=band_mean,
            band_deviation=band_deviation,
        )
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


def _pooled(
    counts: Sequence[int], means: Sequence[float], deviations: Sequence[float]
) -> tuple[float, float]:
    """Return the mean and the deviation of every value the rows were measured over.

    Args:
        counts: How many values each row was measured over.
        means: The mean each row holds.
        deviations: The standard deviation each row holds.

    Returns:
        mean: The mean over all of them together.
        deviation: Their standard deviation, worked from the second moment each
            row carries rather than from the values themselves.
    """
    weight = np.asarray(counts, dtype=float)
    held = np.asarray(means, dtype=float)
    spread = np.asarray(deviations, dtype=float)
    total = weight.sum()
    if not total:
        return (0.0, 0.0)
    mean = float((weight * held).sum() / total)
    second = float((weight * (spread**2 + held**2)).sum() / total)
    return (mean, float(np.sqrt(max(second - mean**2, 0.0))))


def _banded(
    rows: Sequence[ObservationMetadata],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return the mean and the deviation of each band, over the rows that carry them.

    Args:
        rows: The index rows of one instrument, which a build made before the
            per band columns landed carries none of them on.

    Returns:
        mean: The mean of each band, or None where the build published none.
        deviation: The standard deviation of each band, or None for the same
            reason.
    """
    held = [one for one in rows if getattr(one, "band_mean", None) is not None]
    if not held:
        return (None, None)
    bands = len(held[0].band_mean)
    means, deviations = [], []
    for band in range(bands):
        pooled = _pooled(
            [one.band_valid_count[band] for one in held],
            [one.band_mean[band] for one in held],
            [one.band_std[band] for one in held],
        )
        means.append(pooled[0])
        deviations.append(pooled[1])
    return (np.asarray(means), np.asarray(deviations))
