"""One panel per model, laid in rows of a few each."""

from __future__ import annotations

import math
from collections.abc import Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLUMNS = 3


def panel_grid(titles: Sequence[str]) -> go.Figure:
    """Return an empty figure holding one titled panel per model.

    Args:
        titles: The title of each panel, in the order they fill the rows.

    Returns:
        figure: The panels, COLUMNS a row.
    """
    return make_subplots(
        rows=math.ceil(len(titles) / COLUMNS),
        cols=min(len(titles), COLUMNS),
        subplot_titles=list(titles),
        horizontal_spacing=0.08,
        vertical_spacing=0.12 if len(titles) > COLUMNS else 0.08,
    )


def panel_place(at: int) -> dict[str, int]:
    """Return where one panel sits in the grid.

    Args:
        at: Which panel, counting from 0.

    Returns:
        place: Its row and column, as plotly counts them from 1.
    """
    return {"row": at // COLUMNS + 1, "col": at % COLUMNS + 1}


def grid_size(count: int, panel: int) -> dict[str, int]:
    """Return how wide and high a grid of square panels is drawn.

    Args:
        count: How many panels it holds.
        panel: The side of one panel, in pixels.

    Returns:
        size: Its width and height, in pixels.
    """
    return {
        "width": panel * min(count, COLUMNS) + 200,
        "height": panel * math.ceil(count / COLUMNS) + 80,
    }
