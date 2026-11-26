"""Syncing one Leonardo job's offline runs to W&B, then removing what the job wrote."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from dotenv import load_dotenv

from mars_multisensor_mae.configs.paths import ENV_PATH
from mars_multisensor_mae.logs.tracker import sync_offline_run


def sync_leonardo_job(job: Path) -> None:
    """Sync every offline run of one job in the order logged, then remove its folder.

    Args:
        job: The job's folder, its runs under `wandb/`.
    """
    runs = sorted((job / "wandb").glob("offline-run-*"))
    if not runs:
        raise FileNotFoundError(f"{job} holds no offline run")
    for directory in runs:
        sync_offline_run(directory)
    shutil.rmtree(job)


if __name__ == "__main__":
    parsed = argparse.ArgumentParser(description=__doc__)
    parsed.add_argument("job", type=Path, help="the job's folder under $WORK/jobs")
    load_dotenv(ENV_PATH)
    sync_leonardo_job(parsed.parse_args().job)
