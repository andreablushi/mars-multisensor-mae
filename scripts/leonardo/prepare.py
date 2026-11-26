"""Preparing one evaluation on Leonardo's login node: what its offline job reads."""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from mars_multisensor_mae.architecture.load import trained_run
from mars_multisensor_mae.configs.paths import (
    ARTIFACTS_ROOT,
    ENV_PATH,
    results_directory,
)
from mars_multisensor_mae.logs.evaluation import (
    LAVAFLOW,
    RECONSTRUCTION,
    kept_evaluation_run,
)
from mars_multisensor_mae.logs.tracker import downloaded_artifact


def prepare_evaluation(run_name: str, reconstruction: bool, lavaflow: bool) -> None:
    """Download the model an evaluation reads and the parts a part asked alone keeps.

    Args:
        run_name: What the run is called, a last part of "random" reading no model.
        reconstruction: Whether the reconstruction is asked alone.
        lavaflow: Whether the lava flow retrieval is asked alone.
    """
    if trained_run(run_name):
        downloaded_artifact(run_name, ARTIFACTS_ROOT, "model")
    asked = [
        part
        for part, on in ((RECONSTRUCTION, reconstruction), (LAVAFLOW, lavaflow))
        if on
    ]
    if asked:
        kept_evaluation_run(run_name, asked, results_directory(run_name, "kept"))


if __name__ == "__main__":
    parsed = argparse.ArgumentParser(description=__doc__)
    parsed.add_argument("run_name", help="what the evaluated run is called")
    parsed.add_argument(
        "--reconstruction", action="store_true", help="the reconstruction alone"
    )
    parsed.add_argument(
        "--lavaflow", action="store_true", help="the lava flow retrieval alone"
    )
    load_dotenv(ENV_PATH)
    prepare_evaluation(**vars(parsed.parse_args()))
