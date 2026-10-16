"""Scores side by side, one row per model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
from pandas.io.formats.style import Styler

from figures.style import GAIN_COLOR, LOSS_COLOR

FULL_GAIN = 0.5


def comparison_table(
    scores: Mapping[str, Mapping[str, float]], columns: Sequence[str]
) -> pd.DataFrame:
    """Set every model's scores side by side, one row each."""
    return pd.DataFrame.from_dict(scores, orient="index")[list(columns)]


def compared_table(table: pd.DataFrame, baseline: str) -> Styler:
    """Colour every score by its gain on the baseline's, green above."""

    def colours(column: pd.Series) -> list[str]:
        """Return each cell's background, deeper the further from the baseline."""
        if baseline not in column:
            return [""] * len(column)
        return [
            ""
            if model == baseline
            else "background-color: rgba({}, {}, {}, {:.2f})".format(
                *(GAIN_COLOR if gain > 0 else LOSS_COLOR), min(abs(gain) / FULL_GAIN, 1)
            )
            for model, gain in (column - column[baseline]).items()
        ]

    return table.style.apply(colours).format("{:.3f}")
