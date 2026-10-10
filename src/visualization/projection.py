"""Tiles laid on the UMAP plane, one panel per model and view."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from evaluation.metrics import ViewScores
from visualization.style import CLASS_COLORS, CLASS_SYMBOLS, TEMPLATE


def projection_figure(
    scores: Mapping[str, Mapping[str, ViewScores]], classes: Sequence[str]
) -> go.Figure:
    """Lay every model's tiles on the UMAP plane, a row per model, a column per view."""
    models = list(scores)
    views = list(next(iter(scores.values())))
    figure = make_subplots(
        rows=len(models),
        cols=len(views),
        subplot_titles=[f"{model} {view}" for model in models for view in views],
    )
    for row, model in enumerate(models, start=1):
        for column, view in enumerate(views, start=1):
            one = scores[model][view]
            labels = np.asarray(one.labels)
            tiles = np.asarray(one.tiles)
            for at, name in enumerate(classes):
                taken = labels == name
                figure.add_trace(
                    go.Scatter(
                        x=one.projection[taken, 0],
                        y=one.projection[taken, 1],
                        mode="markers",
                        name=name,
                        legendgroup=name,
                        showlegend=row == 1 and column == 1,
                        customdata=tiles[taken],
                        hovertemplate=f"%{{customdata}}<br>{name}<extra></extra>",
                        marker={
                            "size": 7,
                            "color": CLASS_COLORS[at],
                            "symbol": CLASS_SYMBOLS[at],
                            "line": {"width": 1, "color": "white"},
                        },
                    ),
                    row=row,
                    col=column,
                )
    figure.update_xaxes(showticklabels=False)
    figure.update_yaxes(showticklabels=False)
    figure.update_layout(
        template=TEMPLATE,
        height=380 * len(models),
        width=380 * len(views) + 200,
        legend_title="class",
    )
    return figure
