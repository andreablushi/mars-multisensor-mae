"""What each notebook cell shows in one call: the selectors and what they redraw."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import ipywidgets as widgets
import numpy as np
import plotly.graph_objects as go
from IPython.display import display

from configs.paths import RECONSTRUCTION_FILE, RESULTS_FILE, RESULTS_ROOT
from evaluation.evaluate import EVERY_INSTRUMENT, GAP
from evaluation.metrics import TOP_K, compute_metrics
from evaluation.results import Metrics, read_reconstruction, read_tile_vectors
from evaluation.similarity import cosine_similarity
from figures.heatmap import confusion_figure, distance_figure
from figures.projection import projection_figure
from figures.reconstruction import reconstruction_figures
from figures.table import compared_table, comparison_table

BASELINE = "random"
VIEW = "view"
POOLING = "pooling"
NEIGHBOURS = 5
LAYOUT_SEED = 0
RETRIEVAL_ROWS = [
    f"{measure}@{k}" for measure in ("precision", "recall", "f1") for k in TOP_K
]
RANKING_ROWS = ["mAP", *(f"mAP@{k}" for k in TOP_K), "MRR", "R-precision"]
KNN_ROWS = ["accuracy", "balanced accuracy"]
REDRAWS: dict[str, Callable[[], None]] = {}

Scores = Mapping[str, Mapping[tuple[str, str], Metrics]]


def read_scores() -> dict[str, dict[tuple[str, str], Metrics]]:
    """Return every view's metrics of every run fetched into the results directory.

    Returns:
        scores: Each view's metrics, keyed by run, then pooling and view.
    """
    scores = {}
    for path in sorted(RESULTS_ROOT.glob(f"**/{RESULTS_FILE}")):
        tiles, labels, vectors = read_tile_vectors(path)
        scores[path.parent.relative_to(RESULTS_ROOT).as_posix()] = {
            view: compute_metrics(
                cosine_similarity(held), tiles, labels, NEIGHBOURS, LAYOUT_SEED
            )
            for view, held in vectors.items()
        }
    return scores


def selector_controls(scores: Scores) -> dict[str, widgets.ValueWidget]:
    """Show a table of runs to tick, a view and a pooling menu, and Apply.

    Args:
        scores: Each view's metrics, keyed by run, then pooling and view.

    Returns:
        controls: Each run's tick box, keyed by run, the view menu under VIEW and
            the pooling menu under POOLING.
    """
    boxes = {
        run: widgets.Checkbox(value=True, indent=False, layout={"width": "auto"})
        for run in sorted(scores)
        if run != BASELINE
    }
    keys = [key for by_key in scores.values() for key in by_key]
    views = list(dict.fromkeys(view for _, view in keys))
    poolings = sorted({pooling for pooling, _ in keys})
    menus = {
        VIEW: widgets.Dropdown(
            options=views,
            value=EVERY_INSTRUMENT if EVERY_INSTRUMENT in views else views[0],
            description="view",
        ),
        POOLING: widgets.Dropdown(
            options=poolings,
            value=GAP if GAP in poolings else poolings[0],
            description="pooling",
        ),
    }
    apply = widgets.Button(description="Apply", button_style="primary")
    apply.on_click(lambda _: [redraw() for redraw in REDRAWS.values()])
    table = widgets.GridBox(
        [widgets.HTML("<b>show</b>"), widgets.HTML("<b>model</b>")]
        + [cell for run, box in boxes.items() for cell in (box, widgets.Label(run))],
        layout={
            "grid_template_columns": "50px auto",
            "grid_gap": "0 12px",
            "width": "max-content",
        },
    )
    display(widgets.VBox([table, *menus.values(), apply]))
    return boxes | menus


def shown(
    section: str,
    draw: Callable[[dict[str, Metrics], str], widgets.DOMWidget],
    scores: Scores,
    controls: Mapping[str, widgets.ValueWidget],
) -> None:
    """Show what draw makes of the baseline and the ticked runs, redrawn on Apply."""
    shown_box = widgets.Box()

    def redraw() -> None:
        """Swap the section's widget for one of the chosen runs holding the view."""
        key = controls[POOLING].value, controls[VIEW].value
        runs = [BASELINE] + [
            run
            for run, box in controls.items()
            if run not in (VIEW, POOLING) and box.value
        ]
        chosen = {
            run: scores[run][key]
            for run in runs
            if run in scores and key in scores[run]
        }
        shown_box.children = (draw(chosen, " ".join(key)),) if chosen else ()

    # A rerun of the section replaces its earlier redraw, so only one draws
    REDRAWS[section] = redraw
    redraw()
    display(shown_box)


def shown_figure(
    section: str,
    figure: Callable[[dict[str, Metrics]], go.Figure],
    scores: Scores,
    controls: Mapping[str, widgets.ValueWidget],
) -> None:
    """Show one figure of the chosen runs' view."""
    shown(
        section,
        lambda chosen, view: go.FigureWidget(figure(chosen).update_layout(title=view)),
        scores,
        controls,
    )


def shown_table(
    section: str,
    columns: Sequence[str],
    score: Callable[[Metrics], Mapping[str, float]],
    scores: Scores,
    controls: Mapping[str, widgets.ValueWidget],
) -> None:
    """Show the chosen runs' scores of the view, coloured by gain on the baseline."""
    shown(
        section,
        lambda chosen, view: widgets.HTML(
            compared_table(
                comparison_table(
                    {run: score(one) for run, one in chosen.items()}, columns
                ),
                BASELINE,
            ).to_html()
        ),
        scores,
        controls,
    )


def mean_scores(scores: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Return every per tile score averaged over the tiles.

    Args:
        scores: Each score of every tile. (T,)

    Returns:
        means: Each score's mean.
    """
    return {name: float(held.mean()) for name, held in scores.items()}


def show_projection(
    scores: Scores, controls: Mapping[str, widgets.ValueWidget]
) -> None:
    """Show every chosen run's tiles on the UMAP plane."""
    shown_figure("projection", projection_figure, scores, controls)


def show_retrieval(scores: Scores, controls: Mapping[str, widgets.ValueWidget]) -> None:
    """Show the mean retrieval scores."""
    shown_table(
        "retrieval",
        RETRIEVAL_ROWS,
        lambda one: mean_scores(one.retrieval),
        scores,
        controls,
    )


def show_ranking(scores: Scores, controls: Mapping[str, widgets.ValueWidget]) -> None:
    """Show the mean ranking scores."""
    shown_table(
        "ranking",
        RANKING_ROWS,
        lambda one: mean_scores(one.retrieval),
        scores,
        controls,
    )


def show_class_ranking(
    scores: Scores, controls: Mapping[str, widgets.ValueWidget]
) -> None:
    """Show the mAP of each class's tiles."""
    first = next(iter(next(iter(scores.values())).values()))
    shown_table(
        "class ranking",
        first.classes,
        lambda one: {
            name: float(one.retrieval["mAP"][np.asarray(one.labels) == name].mean())
            for name in one.classes
        },
        scores,
        controls,
    )


def show_distances(scores: Scores, controls: Mapping[str, widgets.ValueWidget]) -> None:
    """Show the mean distance between classes."""
    shown_figure("distances", distance_figure, scores, controls)


def show_knn(scores: Scores, controls: Mapping[str, widgets.ValueWidget]) -> None:
    """Show the kNN scores."""
    shown_table("knn", KNN_ROWS, lambda one: one.knn, scores, controls)


def show_confusion(scores: Scores, controls: Mapping[str, widgets.ValueWidget]) -> None:
    """Show where the kNN vote put each class."""
    shown_figure("confusion", confusion_figure, scores, controls)


def show_reconstruction() -> None:
    """Show one run's rebuilt observations, the run picked from a menu."""
    paths = {
        path.parent.relative_to(RESULTS_ROOT).as_posix(): path
        for path in sorted(RESULTS_ROOT.glob(f"**/{RECONSTRUCTION_FILE}"))
    }
    menu = widgets.Dropdown(options=list(paths), description="model")
    shown_box = widgets.VBox()

    def redraw(_: object = None) -> None:
        """Swap the figures for the picked run's."""
        tile, reconstruction = read_reconstruction(paths[menu.value])
        shown_box.children = [widgets.HTML(f"<b>tile {tile}</b>")] + [
            go.FigureWidget(figure) for figure in reconstruction_figures(reconstruction)
        ]

    menu.observe(redraw, names="value")
    if paths:
        redraw()
    display(widgets.VBox([menu, shown_box]))
