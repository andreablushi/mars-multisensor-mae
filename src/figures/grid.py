"""One panel per model, laid in rows of a few each."""

from __future__ import annotations

import math
from collections.abc import Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLUMNS = 3
GAP = 180


def panel_grid(titles: Sequence[str], panel: int) -> go.Figure:
    """Return an empty figure holding one titled square panel per model.

    Args:
        titles: The title of each panel, in the order they fill the rows.
        panel: The side of one panel, in pixels.

    Returns:
        figure: The panels, COLUMNS a row, GAP pixels apart for their labels.
    """
    rows, cols = math.ceil(len(titles) / COLUMNS), min(len(titles), COLUMNS)
    width, height = panel * cols + GAP * (cols - 1), panel * rows + GAP * (rows - 1)
    return make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=list(titles),
        horizontal_spacing=GAP / width,
        vertical_spacing=GAP / height,
    ).update_layout(
        width=width + 200 + GAP // 2,
        height=height + 80,
        # Clear of the last column's slanted labels
        coloraxis_colorbar_x=1 + GAP / 2 / width,
    )


def panel_place(at: int) -> dict[str, int]:
    """Return where one panel sits in the grid.

    Args:
        at: Which panel, counting from 0.

    Returns:
        place: Its row and column, as plotly counts them from 1.
    """
    return {"row": at // COLUMNS + 1, "col": at % COLUMNS + 1}
