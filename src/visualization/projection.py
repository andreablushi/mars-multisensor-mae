"""Tiles laid on the UMAP plane, one panel per model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from evaluation.metrics import ViewScores
from visualization.style import CLASS_COLORS, CLASS_SYMBOLS, TEMPLATE


def projection_figure(
    scores: Mapping[str, ViewScores], classes: Sequence[str]
) -> go.Figure:
    """Lay every model's tiles on the UMAP plane, one panel each, coloured by class."""
    figure = make_subplots(rows=1, cols=len(scores), subplot_titles=list(scores))
    for column, one in enumerate(scores.values(), start=1):
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
                    showlegend=column == 1,
                    customdata=tiles[taken],
                    hovertemplate=f"%{{customdata}}<br>{name}<extra></extra>",
                    marker={
                        "size": 7,
                        "color": CLASS_COLORS[at],
                        "symbol": CLASS_SYMBOLS[at],
                        "line": {"width": 1, "color": "white"},
                    },
                ),
                row=1,
                col=column,
            )
    figure.update_xaxes(showticklabels=False)
    figure.update_yaxes(showticklabels=False)
    figure.update_layout(
        template=TEMPLATE,
        height=420,
        width=400 * len(scores) + 200,
        legend_title="class",
    )
    return figure
