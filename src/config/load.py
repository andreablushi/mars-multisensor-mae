"""Composing the configs of one run, and reading them under the schema."""

from __future__ import annotations

from collections.abc import Sequence

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from config.paths import CONFIGS_ROOT
from config.schema import Config

ROOT_CONFIG_NAME = "config"


def load_config(overrides: Sequence[str] = ()) -> Config:
    """Return what one run is settled from, composed and read under the schema.

    Args:
        overrides: What to compose it with instead of the defaults, as hydra
            spells them: `dataset=full` for another file of a group,
            `dataset.seed=7` for one value of one.

    Returns:
        config: The run's choices, every key checked against the schema, so a
            key the schema does not declare is an error rather than a line that
            settles nothing.
    """
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS_ROOT)):
        composed = compose(config_name=ROOT_CONFIG_NAME, overrides=list(overrides))
    return OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Config), composed))
