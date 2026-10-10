"""What each notebook cell shows in one call: the selectors and what they redraw."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import ipywidgets as widgets
from IPython.display import display

from configs.paths import RECONSTRUCTION_FILE, RESULTS_ROOT, reconstruction_path
from evaluation.metrics import TOP_K, ViewScores, scores_by_run
from evaluation.results import Reconstruction, read_reconstructions
from evaluation.similarity import EVERY_INSTRUMENT
from visualization.heatmap import confusion_figure, distance_figure
from visualization.projection import projection_figure
from visualization.reconstruction import reconstruction_figure
from visualization.retrieval import f1_figure
from visualization.table import compared_table, comparison_table

BASELINE = "random"
NEIGHBOURS = 5
LAYOUT_SEED = 0
K = 10
RETRIEVAL_ROWS = [
    f"{measure}@{k}" for measure in ("precision", "recall", "f1") for k in TOP_K
]
KNN_ROWS = ["accuracy", "balanced accuracy"]

Scores = Mapping[str, Mapping[str, ViewScores]]


def read_scores() -> dict[str, dict[str, ViewScores]]:
    """Return every view's scores of every run fetched into the results directory.

    Returns:
        scores: Each view's scores, keyed by run, then view.
    """
    return scores_by_run(RESULTS_ROOT, NEIGHBOURS, LAYOUT_SEED)


def model_selector(runs: Sequence[str]) -> dict[str, widgets.Checkbox]:
    """Show one tick box per run, every one ticked, and hand them back.

    Args:
        runs: The runs to choose among, the baseline put first.

    Returns:
        models: Each run's tick box, keyed by run.
    """
    order = sorted(runs, key=lambda run: (run != BASELINE, run))
    models = {
        run: widgets.Checkbox(value=True, description=run, indent=False)
        for run in order
    }
    display(widgets.HBox(list(models.values())))
    return models


def shown(
    draw: Callable[[list[str]], None], models: Mapping[str, widgets.Checkbox]
) -> None:
    """Show what draw makes of the ticked runs, redrawn whenever a tick changes."""

    def ticked(**held: bool) -> None:
        """Draw the ticked runs, if any."""
        chosen = [run for run, kept in held.items() if kept]
        if chosen:
            draw(chosen)

    display(widgets.interactive_output(ticked, dict(models)))


def show_projection(scores: Scores, models: Mapping[str, widgets.Checkbox]) -> None:
    """Show every ticked run's tiles on the UMAP plane, over every instrument."""
    classes = sorted(set(next(iter(scores.values()))[EVERY_INSTRUMENT].labels))
    shown(
        lambda chosen: projection_figure(
            {run: scores[run][EVERY_INSTRUMENT] for run in chosen}, classes
        ).show(),
        models,
    )


def show_retrieval(scores: Scores, models: Mapping[str, widgets.Checkbox]) -> None:
    """Show the mean retrieval scores, coloured by their gain on the baseline."""
    shown(
        lambda chosen: display(
            compared_table(
                comparison_table(
                    {
                        run: {
                            view: {
                                name: score.mean()
                                for name, score in one.retrieval.items()
                            }
                            for view, one in scores[run].items()
                        }
                        for run in chosen
                    },
                    RETRIEVAL_ROWS,
                ),
                BASELINE,
            )
        ),
        models,
    )


def show_f1(scores: Scores, models: Mapping[str, widgets.Checkbox]) -> None:
    """Show every query tile's F1 at K over every instrument."""
    shown(
        lambda chosen: f1_figure(
            {run: scores[run][EVERY_INSTRUMENT] for run in chosen}, K
        ).show(),
        models,
    )


def show_distances(scores: Scores, models: Mapping[str, widgets.Checkbox]) -> None:
    """Show the mean distance between classes over every instrument."""
    shown(
        lambda chosen: distance_figure(
            {run: scores[run][EVERY_INSTRUMENT] for run in chosen}
        ).show(),
        models,
    )


def show_knn(scores: Scores, models: Mapping[str, widgets.Checkbox]) -> None:
    """Show the kNN scores, coloured by their gain on the baseline."""
    shown(
        lambda chosen: display(
            compared_table(
                comparison_table(
                    {
                        run: {view: one.knn for view, one in scores[run].items()}
                        for run in chosen
                    },
                    KNN_ROWS,
                ),
                BASELINE,
            )
        ),
        models,
    )


def show_confusion(scores: Scores, models: Mapping[str, widgets.Checkbox]) -> None:
    """Show where the kNN vote put each class over every instrument."""
    shown(
        lambda chosen: confusion_figure(
            {run: scores[run][EVERY_INSTRUMENT] for run in chosen}
        ).show(),
        models,
    )


def read_reconstructions_by_run() -> dict[str, dict[str, dict[str, Reconstruction]]]:
    """Return every reconstructed tile of every run fetched into the results directory.

    Returns:
        reconstructions: Each instrument's reconstruction, keyed by run, tile, then
            instrument.
    """
    return {
        path.parent.name: read_reconstructions(reconstruction_path(path.parent.name))
        for path in sorted(RESULTS_ROOT.glob(f"*/{RECONSTRUCTION_FILE}"))
    }


def reconstruction_selector(
    reconstructions: Mapping[str, Mapping[str, object]],
) -> tuple[widgets.Dropdown, widgets.Dropdown]:
    """Show a run and a tile to pick, and hand both back.

    Args:
        reconstructions: Each run's reconstructed tiles.

    Returns:
        model: The run picked.
        tile: The tile picked.
    """
    runs = sorted(reconstructions)
    model = widgets.Dropdown(options=runs, description="model")
    tile = widgets.Dropdown(
        options=sorted(reconstructions[runs[0]]), description="tile"
    )
    display(widgets.HBox([model, tile]))
    return model, tile


def show_reconstruction(
    reconstructions: Mapping[str, Mapping[str, Mapping[str, Reconstruction]]],
    model: widgets.Dropdown,
    tile: widgets.Dropdown,
    instrument: str,
) -> None:
    """Show one instrument of the picked run and tile, redrawn on every pick."""

    def draw(model: str, tile: str) -> None:
        """Draw the instrument read, hidden and written back, if the tile holds it."""
        held = reconstructions[model][tile]
        if instrument in held:
            reconstruction_figure(held[instrument], f"{model} {tile}").show()

    display(widgets.interactive_output(draw, {"model": model, "tile": tile}))
