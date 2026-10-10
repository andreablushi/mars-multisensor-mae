"""Class against class matrices, one panel per model."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from evaluation.metrics import ViewScores
from visualization.style import BLUES, OUTLINE_COLOR, TEMPLATE


def distance_figure(scores: Mapping[str, ViewScores]) -> go.Figure:
    """Draw the mean distance between classes, each row's nearest class outlined."""
    figure = make_subplots(rows=1, cols=len(scores), subplot_titles=list(scores))
    for column, one in enumerate(scores.values(), start=1):
        figure.add_trace(
            go.Heatmap(
                z=one.distances,
                x=one.classes,
                y=one.classes,
                colorscale=BLUES,
                reversescale=True,
                coloraxis="coloraxis",
                text=np.round(one.distances, 3),
                texttemplate="%{text}",
                hovertemplate="%{y} to %{x}: %{z:.3f}<extra></extra>",
            ),
            row=1,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=[one.classes[at] for at in one.distances.argmin(axis=1)],
                y=one.classes,
                mode="markers",
                showlegend=False,
                hoverinfo="skip",
                marker={
                    "symbol": "square-open",
                    "size": 34,
                    "color": OUTLINE_COLOR,
                    "line": {"width": 3},
                },
            ),
            row=1,
            col=column,
        )
    figure.update_yaxes(autorange="reversed")
    figure.update_layout(
        template=TEMPLATE,
        coloraxis={
            "colorscale": BLUES,
            "reversescale": True,
            "colorbar": {"title": "mean distance"},
        },
        height=560,
        width=560 * len(scores) + 120,
    )
    return figure


def confusion_figure(scores: Mapping[str, ViewScores]) -> go.Figure:
    """Draw how many tiles of each class the kNN vote put into each."""
    figure = make_subplots(rows=1, cols=len(scores), subplot_titles=list(scores))
    for column, one in enumerate(scores.values(), start=1):
        figure.add_trace(
            go.Heatmap(
                z=one.confusion,
                x=one.classes,
                y=one.classes,
                coloraxis="coloraxis",
                text=one.confusion,
                texttemplate="%{text}",
                hovertemplate="%{y} voted %{x}: %{z}<extra></extra>",
            ),
            row=1,
            col=column,
        )
    figure.update_yaxes(autorange="reversed", title_text="class")
    figure.update_xaxes(title_text="voted")
    figure.update_layout(
        template=TEMPLATE,
        coloraxis={"colorscale": BLUES, "colorbar": {"title": "tiles"}},
        height=560,
        width=560 * len(scores) + 120,
    )
    return figure
