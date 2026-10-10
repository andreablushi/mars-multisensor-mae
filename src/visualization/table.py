"""Scores side by side, one column per view and model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd


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


def gain_table(table: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Show every model's scores less the baseline's in the same view."""
    gains = {
        (view, model): table[view, model] - table[view, baseline]
        for view, model in table.columns
        if model != baseline
    }
    return pd.DataFrame(gains)
