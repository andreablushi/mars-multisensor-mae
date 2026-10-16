"""Tiles laid on the UMAP plane, one panel per model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import plotly.graph_objects as go

from evaluation.results import Metrics
from figures.grid import panel_grid, panel_place
from figures.style import CLASS_COLORS, CLASS_SYMBOLS, TEMPLATE

PANEL = 400


def projection_figure(
    scores: Mapping[str, Metrics], classes: Sequence[str]
) -> go.Figure:
    """Lay every model's tiles on the UMAP plane, one panel each, coloured by class."""
    figure = panel_grid(list(scores), PANEL)
    for at, one in enumerate(scores.values()):
        labels = np.asarray(one.labels)
        tiles = np.asarray(one.tiles)
        for order, name in enumerate(classes):
            taken = labels == name
            figure.add_trace(
                go.Scatter(
                    x=one.projection[taken, 0],
                    y=one.projection[taken, 1],
                    mode="markers",
                    name=name,
                    legendgroup=name,
                    showlegend=at == 0,
                    customdata=tiles[taken],
                    hovertemplate=f"%{{customdata}}<br>{name}<extra></extra>",
                    marker={
                        "size": 7,
                        "color": CLASS_COLORS[order],
                        "symbol": CLASS_SYMBOLS[order],
                        "line": {"width": 1, "color": "white"},
                    },
                ),
                **panel_place(at),
            )
    figure.update_xaxes(showticklabels=False)
    figure.update_yaxes(showticklabels=False)
    figure.update_layout(template=TEMPLATE, legend_title="class")
    return figure
