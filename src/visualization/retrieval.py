"""Each query tile's F1 at k, one row per model, tiles grouped by class."""

from __future__ import annotations

from collections.abc import Mapping

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from evaluation.metrics import ViewScores
from visualization.style import BAND_COLOR, F1_COLOR, TEMPLATE


def f1_figure(scores: Mapping[str, ViewScores], k: int) -> go.Figure:
    """Draw every query tile's F1 at k, a row per model, tiles grouped by class."""
    first = next(iter(scores.values()))
    order = sorted(
        range(len(first.tiles)), key=lambda at: (first.labels[at], first.tiles[at])
    )
    tiles = [first.tiles[at] for at in order]
    labels = [first.labels[at] for at in order]
    bounds = [at for at in range(1, len(labels)) if labels[at] != labels[at - 1]]
    starts, ends = [0, *bounds], [*bounds, len(labels)]
    figure = make_subplots(
        rows=len(scores),
        cols=1,
        shared_xaxes=True,
        subplot_titles=list(scores),
        vertical_spacing=0.08,
    )
    for row, one in enumerate(scores.values(), start=1):
        where = {tile: at for at, tile in enumerate(one.tiles)}
        held = [where.get(tile) for tile in tiles]
        figure.add_trace(
            go.Bar(
                x=list(range(len(tiles))),
                y=[None if at is None else one.retrieval[f"f1@{k}"][at] for at in held],
                showlegend=False,
                marker={"color": F1_COLOR, "line": {"width": 0}},
                customdata=list(zip(tiles, labels, strict=True)),
                hovertemplate=(
                    f"%{{customdata[0]}}<br>%{{customdata[1]}}<br>f1@{k} "
                    "%{y:.2f}<extra></extra>"
                ),
            ),
            row=row,
            col=1,
        )
    figure.update_xaxes(
        tickvals=[
            (start + end - 1) / 2 for start, end in zip(starts, ends, strict=True)
        ],
        ticktext=[labels[start] for start in starts],
        tickangle=-30,
        range=[-0.5, len(tiles) - 0.5],
    )
    for start, end in list(zip(starts, ends, strict=True))[1::2]:
        figure.add_vrect(
            x0=start - 0.5,
            x1=end - 0.5,
            fillcolor=BAND_COLOR,
            line_width=0,
            layer="below",
            row="all",
            col=1,
        )
    figure.update_yaxes(range=[0, 1], title_text=f"f1@{k}")
    figure.update_layout(
        template=TEMPLATE,
        height=160 + 260 * len(scores),
        width=250 + 4 * len(tiles),
        bargap=0.15,
    )
    figure.update_xaxes(
        title_text=f"query tile, by class ({len(tiles)})", row=len(scores)
    )
    return figure
