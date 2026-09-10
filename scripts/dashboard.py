"""Laying out the Weights & Biases workspace, training beside validation."""

from __future__ import annotations

import os

import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws
from dotenv import load_dotenv

from config.load import load_config
from config.paths import REPO_ROOT

VIEW = "Training"
STEP = "Step"
RETRIEVAL = {
    "precision_at_k": "Precision at k",
    "mean_average_precision": "Mean average precision",
    "nearest_neighbour_accuracy": "Nearest neighbour accuracy",
}


def paired_plot(title: str, name: str) -> wr.LinePlot:
    """Return one line plot of a loss term over training beside it over validation.

    Args:
        title: What the plot is called.
        name: The term, as the loss names it under "train/" and "validation/".

    Returns:
        plot: Both lines against the training step.
    """
    return wr.LinePlot(title=title, x=STEP, y=[f"train/{name}", f"validation/{name}"])


def main() -> None:
    """Save the workspace view, laid out for the instruments the model reads."""
    load_dotenv(REPO_ROOT / ".env")
    instruments = load_config().model.instruments
    sections = [
        ws.Section(
            name="Loss",
            is_open=True,
            panels=[
                paired_plot("Total", "loss"),
                paired_plot("Uniformity", "uniformity"),
            ],
        ),
        ws.Section(
            name="Reconstruction",
            is_open=True,
            panels=[
                paired_plot(name, f"reconstruction/{name}") for name in instruments
            ],
        ),
        ws.Section(
            name="Cross-modal",
            is_open=True,
            panels=[paired_plot(name, f"cross/{name}") for name in instruments],
        ),
        ws.Section(
            name="Retrieval",
            is_open=True,
            panels=[
                wr.LinePlot(title=title, x=STEP, y=[f"validation/{name}"])
                for name, title in RETRIEVAL.items()
            ],
        ),
        ws.Section(
            name="Optimisation",
            is_open=True,
            panels=[wr.LinePlot(title="Learning rate", x=STEP, y=["learning_rate"])],
        ),
    ]
    workspace = ws.Workspace(
        entity=os.environ["WANDB_ENTITY"],
        project=os.environ["WANDB_PROJECT"],
        name=VIEW,
        sections=sections,
    )
    workspace.save()
    print(workspace.url)


if __name__ == "__main__":
    main()
