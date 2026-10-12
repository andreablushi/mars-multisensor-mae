"""What each notebook cell shows in one call: the selectors and what they redraw."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import ipywidgets as widgets
import plotly.graph_objects as go
from IPython.display import display

from configs.paths import RESULTS_FILE, RESULTS_ROOT
from evaluation.evaluate import EVERY_INSTRUMENT
from evaluation.metrics import TOP_K, compute_metrics
from evaluation.results import Metrics, read_tile_vectors
from evaluation.similarity import cosine_similarity
from figures.heatmap import confusion_figure, distance_figure
from figures.projection import projection_figure
from figures.retrieval import f1_figure
from figures.table import compared_table, comparison_table

BASELINE = "random"
VIEW = "view"
NEIGHBOURS = 5
LAYOUT_SEED = 0
K = 10
RETRIEVAL_ROWS = [
    f"{measure}@{k}" for measure in ("precision", "recall", "f1") for k in TOP_K
]
KNN_ROWS = ["accuracy", "balanced accuracy"]
REDRAWS: dict[str, tuple[list[widgets.ValueWidget], Callable[[object], None]]] = {}

Scores = Mapping[str, Mapping[str, Metrics]]


def read_scores() -> dict[str, dict[str, Metrics]]:
    """Return every view's metrics of every run fetched into the results directory.

    Returns:
        scores: Each view's metrics, keyed by run, then view.
    """
    scores = {}
    for path in sorted(RESULTS_ROOT.glob(f"*/{RESULTS_FILE}")):
        tiles, labels, vectors = read_tile_vectors(path)
        scores[path.parent.name] = {
            view: compute_metrics(
                cosine_similarity(held), tiles, labels, NEIGHBOURS, LAYOUT_SEED
            )
            for view, held in vectors.items()
        }
    return scores


def selector_controls(scores: Scores) -> dict[str, widgets.ValueWidget]:
    """Show one tick box per run, every one ticked, and a view menu, and hand them back.

    Args:
        scores: Each view's metrics, keyed by run, then view.

    Returns:
        controls: Each run's tick box, keyed by run, and the view menu under VIEW.
    """
    order = sorted(scores, key=lambda run: (run != BASELINE, run))
    controls = {
        run: widgets.Checkbox(value=True, description=run, indent=False)
        for run in order
    }
    views = list(dict.fromkeys(view for by_view in scores.values() for view in by_view))
    menu = widgets.Dropdown(
        options=views,
        value=EVERY_INSTRUMENT if EVERY_INSTRUMENT in views else views[0],
        description="view",
    )
    display(widgets.VBox([menu, widgets.HBox(list(controls.values()))]))
    return controls | {VIEW: menu}


def shown(
    section: str,
    draw: Callable[[dict[str, Metrics], str], widgets.DOMWidget],
    scores: Scores,
    controls: Mapping[str, widgets.ValueWidget],
) -> None:
    """Show what draw makes of the ticked runs' chosen view, redrawn on every change."""
    shown_box = widgets.Box()

    def redraw(_: object = None) -> None:
        """Swap the section's widget for one of the ticked runs holding the view."""
        view = controls[VIEW].value
        chosen = {
            run: scores[run][view]
            for run, box in controls.items()
            if run != VIEW and box.value and view in scores[run]
        }
        shown_box.children = (draw(chosen, view),) if chosen else ()

    # A rerun of the section replaces its earlier redraw, so only one draws
    boxes, earlier = REDRAWS.get(section, ([], None))
    for box in boxes:
        box.unobserve(earlier, names="value")
    for box in controls.values():
        box.observe(redraw, names="value")
    REDRAWS[section] = (list(controls.values()), redraw)
    redraw()
    display(shown_box)


def shown_figure(
    section: str,
    figure: Callable[[dict[str, Metrics]], go.Figure],
    scores: Scores,
    controls: Mapping[str, widgets.ValueWidget],
) -> None:
    """Show one figure of the ticked runs' chosen view, redrawn on every change."""
    shown(
        section,
        lambda chosen, view: go.FigureWidget(figure(chosen).update_layout(title=view)),
        scores,
        controls,
    )


def shown_table(
    section: str,
    rows: Sequence[str],
    score: Callable[[Metrics], Mapping[str, float]],
    scores: Scores,
    controls: Mapping[str, widgets.ValueWidget],
) -> None:
    """Show the ticked runs' scores of the view, coloured by gain on the baseline."""
    shown(
        section,
        lambda chosen, view: widgets.HTML(
            compared_table(
                comparison_table(
                    {run: {view: score(one)} for run, one in chosen.items()}, rows
                ),
                BASELINE,
            ).to_html()
        ),
        scores,
        controls,
    )


def show_projection(
    scores: Scores, controls: Mapping[str, widgets.ValueWidget]
) -> None:
    """Show every ticked run's tiles on the UMAP plane."""
    first = next(iter(next(iter(scores.values())).values()))
    classes = sorted(set(first.labels))
    shown_figure(
        "projection", lambda one: projection_figure(one, classes), scores, controls
    )


def show_retrieval(scores: Scores, controls: Mapping[str, widgets.ValueWidget]) -> None:
    """Show the mean retrieval scores, coloured by their gain on the baseline."""
    shown_table(
        "retrieval",
        RETRIEVAL_ROWS,
        lambda one: {name: score.mean() for name, score in one.retrieval.items()},
        scores,
        controls,
    )


def show_f1(scores: Scores, controls: Mapping[str, widgets.ValueWidget]) -> None:
    """Show every query tile's F1 at K."""
    shown_figure("f1", lambda one: f1_figure(one, K), scores, controls)


def show_distances(scores: Scores, controls: Mapping[str, widgets.ValueWidget]) -> None:
    """Show the mean distance between classes."""
    shown_figure("distances", distance_figure, scores, controls)


def show_knn(scores: Scores, controls: Mapping[str, widgets.ValueWidget]) -> None:
    """Show the kNN scores, coloured by their gain on the baseline."""
    shown_table("knn", KNN_ROWS, lambda one: one.knn, scores, controls)


def show_confusion(scores: Scores, controls: Mapping[str, widgets.ValueWidget]) -> None:
    """Show where the kNN vote put each class."""
    shown_figure("confusion", confusion_figure, scores, controls)
