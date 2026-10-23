"""What each notebook cell shows in one call: the selectors and what they redraw."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import ipywidgets as widgets
import numpy as np
import plotly.graph_objects as go
from IPython.display import display

from configs.paths import RECONSTRUCTION_FILE, RESULTS_FILE, RESULTS_ROOT
from evaluation.evaluate import EVERY_INSTRUMENT
from evaluation.metrics import compute_metrics
from evaluation.results import Metrics, read_reconstruction, read_tile_vectors
from evaluation.similarity import GAP, cosine_similarity
from figures.heatmap import confusion_figure, distance_figure
from figures.projection import projection_figure
from figures.reconstruction import reconstruction_figures
from figures.table import compared_table, comparison_table

BASELINE = "random"
VIEW = "view"
POOLING = "pooling"
BOTH = "both poolings"
NEIGHBOURS = 5
LAYOUT_SEED = 0
RETRIEVAL_ROWS = ["mAP"]
SUMMARY_ROWS = ["accuracy", "balanced accuracy", "silhouette"]
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
    """Show a table of runs to tick, a view and a pooling menu, a pooling box, Apply.

    Args:
        scores: Each view's metrics, keyed by run, then pooling and view.

    Returns:
        controls: Each run's tick box, keyed by run, the view menu under VIEW, the
            pooling menu under POOLING and the box showing every pooling under BOTH.
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
        BOTH: widgets.Checkbox(value=False, description="compare GAP and CLS"),
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


def labelled(run: str, pooling: str, both: bool) -> str:
    """Return the name a run's row goes by, with its pooling when both are shown.

    Args:
        run: The run.
        pooling: Which pooling its vectors were read with.
        both: Whether every pooling is shown, each as a model of its own.

    Returns:
        name: The run, or the run and its pooling.
    """
    return f"{run} ({pooling})" if both else run


def ticked_runs(controls: Mapping[str, widgets.ValueWidget]) -> list[str]:
    """Return the baseline and every run whose box is ticked.

    Args:
        controls: Each run's tick box, keyed by run, beside the view and pooling menus.

    Returns:
        runs: The baseline first, then the ticked runs.
    """
    return [BASELINE] + [
        run
        for run, box in controls.items()
        if run not in (VIEW, POOLING, BOTH) and box.value
    ]


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
        view, both = controls[VIEW].value, controls[BOTH].value
        poolings = controls[POOLING].options if both else [controls[POOLING].value]
        chosen = {
            labelled(run, pooling, both): scores[run][pooling, view]
            for run in ticked_runs(controls)
            for pooling in poolings
            if run in scores and (pooling, view) in scores[run]
        }
        title = view if both else f"{controls[POOLING].value} {view}"
        shown_box.children = (draw(chosen, title),) if chosen else ()

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

    def table(chosen: dict[str, Metrics], _: str) -> widgets.HTML:
        """Return the scores coloured against the baseline of the menu's pooling."""
        untrained = [
            name for name in chosen if name.split(" (")[0].split("/")[-1] == BASELINE
        ]
        return widgets.HTML(
            compared_table(
                comparison_table(
                    {run: score(one) for run, one in chosen.items()}, columns
                ),
                labelled(BASELINE, controls[POOLING].value, controls[BOTH].value),
                untrained,
            ).to_html()
        )

    shown(section, table, scores, controls)


def mean_scores(scores: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Return every per tile score averaged over the tiles.

    Args:
        scores: Each score of every tile. (T,)

    Returns:
        means: Each score's mean.
    """
    return {name: float(held.mean()) for name, held in scores.items()}


def show_overview(scores: Scores, controls: Mapping[str, widgets.ValueWidget]) -> None:
    """Show the ticked runs' mAP and silhouette of every view, one table per pooling."""
    shown_box = widgets.VBox()
    overall = {
        "mAP": lambda one: one.retrieval["mAP"],
        "silhouette": lambda one: one.silhouette,
    }

    def redraw() -> None:
        """Swap the tables for the baseline's and the ticked runs'."""
        chosen = {run: scores[run] for run in ticked_runs(controls) if run in scores}
        keys = {key for by_key in chosen.values() for key in by_key}
        views = sorted(
            {view for _, view in keys}, key=lambda one: (one == EVERY_INSTRUMENT, one)
        )
        poolings = sorted({pooling for pooling, _ in keys}, key=lambda one: one != GAP)
        untrained = [run for run in chosen if run.split("/")[-1] == BASELINE]
        tables = []
        for name, score in overall.items():
            for pooling in poolings:
                table = comparison_table(
                    {
                        run: {
                            view: float(score(one).mean())
                            for (held, view), one in by_key.items()
                            if held == pooling
                        }
                        for run, by_key in chosen.items()
                    },
                    views,
                )
                tables += [
                    widgets.HTML(f"<b>{pooling} {name}</b>"),
                    widgets.HTML(compared_table(table, BASELINE, untrained).to_html()),
                ]
        shown_box.children = tables

    REDRAWS["overview"] = redraw
    redraw()
    display(shown_box)


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


def show_distances(scores: Scores, controls: Mapping[str, widgets.ValueWidget]) -> None:
    """Show the mean distance between classes."""
    shown_figure("distances", distance_figure, scores, controls)


def show_summary(scores: Scores, controls: Mapping[str, widgets.ValueWidget]) -> None:
    """Show the kNN scores and the mean silhouette side by side."""
    shown_table(
        "summary",
        SUMMARY_ROWS,
        lambda one: one.knn | {"silhouette": float(one.silhouette.mean())},
        scores,
        controls,
    )


def show_confusion(scores: Scores, controls: Mapping[str, widgets.ValueWidget]) -> None:
    """Show where the kNN vote put each class."""
    shown_figure("confusion", confusion_figure, scores, controls)


def show_reconstruction() -> None:
    """Show one run's rebuilt observations, a second run's beside each when picked."""
    paths = {
        path.parent.relative_to(RESULTS_ROOT).as_posix(): path
        for path in sorted(RESULTS_ROOT.glob(f"**/{RECONSTRUCTION_FILE}"))
    }
    menu = widgets.Dropdown(options=list(paths), description="model")
    other = widgets.Dropdown(options=["", *paths], value="", description="compare")
    shown_box = widgets.VBox()

    def redraw(_: object = None) -> None:
        """Swap the figures for the picked runs', each image's rows run by run."""
        runs = list(dict.fromkeys(run for run in (menu.value, other.value) if run))
        tiles, drawn = {}, {}
        for run in runs:
            tiles[run], reconstruction = read_reconstruction(paths[run])
            drawn[run] = reconstruction_figures(reconstruction, run)
        titles = dict.fromkeys(title for run in runs for title in drawn[run])
        shown_box.children = [
            widgets.HTML(
                "<br>".join(f"<b>{run}: tile {tile}</b>" for run, tile in tiles.items())
            )
        ] + [
            go.FigureWidget(drawn[run][title])
            for title in titles
            for run in runs
            if title in drawn[run]
        ]

    menu.observe(redraw, names="value")
    other.observe(redraw, names="value")
    if paths:
        redraw()
    display(widgets.VBox([widgets.HBox([menu, other]), shown_box]))
