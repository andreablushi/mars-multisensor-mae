"""Training one run against the published build, and publishing what it learnt."""

from __future__ import annotations

import sys
from collections.abc import Sequence

import torch
from dh.publish import publish_checkpoint
from dh.store import published_build
from torch.utils.data import DataLoader

from architecture.mae import CrossSensorMAE
from config.load import load_config
from logs.tracker import start_run
from training.data import FeatureDataset, axes_by_instrument, collate, token_shapes
from training.train import train

TRAINING_SPLIT = "train"
VALIDATION_SPLIT = "validation"


def main(overrides: Sequence[str]) -> None:
    """Train one run, as the config composed with the overrides describes it.

    Args:
        overrides: What to compose the config with, as hydra spells them.
    """
    config = load_config(overrides)
    build = published_build(config.dataset)
    by_feature = build.read_observation_metadata_by_feature()
    splits = build.split_features(config.dataset.split, config.dataset.seed)
    statistics = build.compute_stats(set(splits[TRAINING_SPLIT]))
    axes = axes_by_instrument([row for rows in by_feature.values() for row in rows])

    def loader(split: str, seed: int | None) -> DataLoader:
        """Return one split in batches, its features drawn afresh or fixed."""
        features = [by_feature[identity] for identity in splits[split]]
        held = FeatureDataset(build, features, axes, config, statistics, seed)
        return DataLoader(
            held,
            batch_size=config.training.batch_size,
            shuffle=seed is None,
            num_workers=config.training.workers,
            collate_fn=collate,
        )

    training = loader(TRAINING_SPLIT, None)
    validation = loader(VALIDATION_SPLIT, config.dataset.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossSensorMAE(token_shapes(axes, config), config.model).to(device)
    run = start_run(config)
    best = train(model, training, validation, config, device, run)
    run.finish()
    publish_checkpoint(best, f"model-{config.model.name}")


if __name__ == "__main__":
    main(sys.argv[1:])
