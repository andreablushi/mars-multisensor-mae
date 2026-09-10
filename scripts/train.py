"""Training one run against the published build, and publishing what it learnt."""

from __future__ import annotations

import sys
from collections.abc import Sequence

import torch
from dh.publish import publish_checkpoint
from dh.store import published_build

from architecture.mae import CrossSensorMAE
from config.load import load_config
from dataset.draw import split_loaders
from logs.tracker import start_run
from training.train import train


def main(overrides: Sequence[str]) -> None:
    """Train one run, as the config composed with the overrides describes it.

    Args:
        overrides: What to compose the config with, as hydra spells them.
    """
    config = load_config(overrides)
    build = published_build(config.dataset)
    training, validation, shapes = split_loaders(build, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossSensorMAE(shapes, config.model).to(device)
    run = start_run(config)
    best = train(model, training, validation, config, device, run)
    run.finish()
    publish_checkpoint(best, f"model-{config.model.name}")


if __name__ == "__main__":
    main(sys.argv[1:])
