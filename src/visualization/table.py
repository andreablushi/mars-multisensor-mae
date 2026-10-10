"""Scores side by side, one column per view and model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
from pandas.io.formats.style import Styler

from visualization.style import GAIN_COLOR, LOSS_COLOR

FULL_GAIN = 0.5


def comparison_table(
    scores: Mapping[str, Mapping[str, Mapping[str, float]]], rows: Sequence[str]
) -> pd.DataFrame:
    """Set every model's scores side by side, grouped by view."""
    views = next(iter(scores.values()))
    return pd.DataFrame(
        {
            (view, model): by_view[view]
            for view in views
            for model, by_view in scores.items()
        }
    ).loc[list(rows)]


def compared_table(table: pd.DataFrame, baseline: str) -> Styler:
    """Colour every score by its gain on the baseline in the same view, green above."""

    def colours(column: pd.Series) -> list[str]:
        """Return each cell's background, deeper the further from the baseline."""
        view, model = column.name
        if model == baseline or (view, baseline) not in table:
            return [""] * len(column)
        return [
            "background-color: rgba({}, {}, {}, {:.2f})".format(
                *(GAIN_COLOR if gain > 0 else LOSS_COLOR), min(abs(gain) / FULL_GAIN, 1)
            )
            for gain in column - table[view, baseline]
        ]

    return table.style.apply(colours).format("{:.3f}")
