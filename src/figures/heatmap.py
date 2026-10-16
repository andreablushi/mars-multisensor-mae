"""Class against class matrices, one panel per model."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import plotly.graph_objects as go

from evaluation.results import Metrics
from figures.grid import grid_size, panel_grid, panel_place
from figures.style import BLUES, OUTLINE_COLOR, TEMPLATE

PANEL = 420


def distance_figure(scores: Mapping[str, Metrics]) -> go.Figure:
    """Draw the mean distance between classes, each row's nearest class outlined."""
    figure = panel_grid(list(scores))
    for at, one in enumerate(scores.values()):
        figure.add_trace(
            go.Heatmap(
                z=one.distances,
                x=one.classes,
                y=one.classes,
                coloraxis="coloraxis",
                text=np.round(one.distances, 3),
                texttemplate="%{text}",
                hovertemplate="%{y} to %{x}: %{z:.3f}<extra></extra>",
            ),
            **panel_place(at),
        )
        for row, nearest in enumerate(one.distances.argmin(axis=1)):
            figure.add_shape(
                type="rect",
                x0=nearest - 0.5,
                x1=nearest + 0.5,
                y0=row - 0.5,
                y1=row + 0.5,
                line={"color": OUTLINE_COLOR, "width": 3},
                **panel_place(at),
            )
    figure.update_yaxes(autorange="reversed")
    figure.update_layout(
        template=TEMPLATE,
        coloraxis={
            "colorscale": BLUES,
            "reversescale": True,
            "colorbar": {"title": "mean distance"},
        },
        **grid_size(len(scores), PANEL),
    )
    return figure


def confusion_figure(scores: Mapping[str, Metrics]) -> go.Figure:
    """Draw how many tiles of each class the kNN vote put into each."""
    figure = panel_grid(list(scores))
    for at, one in enumerate(scores.values()):
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
            **panel_place(at),
        )
    figure.update_yaxes(autorange="reversed", title_text="class")
    figure.update_xaxes(title_text="voted")
    figure.update_layout(
        template=TEMPLATE,
        coloraxis={"colorscale": BLUES, "colorbar": {"title": "tiles"}},
        **grid_size(len(scores), PANEL),
    )
    return figure
