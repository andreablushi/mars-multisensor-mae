"""Reading the configs of one run under the schema, and the seeds it splits into."""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from dataclasses import fields

from omegaconf import OmegaConf

from configs.paths import CONFIGS_ROOT
from configs.schema import Config


def load_config(overrides: Sequence[str] = ()) -> Config:
    """Return what one run is settled from, read under the schema.

    Args:
        overrides: What to change, as dotted keys: `training.learning_rate=1e-3`.

    Returns:
        config: The run's choices, every key checked against the schema.
    """
    read = {
        one.name: OmegaConf.load(CONFIGS_ROOT / f"{one.name}.yaml")
        for one in fields(Config)
    }
    return OmegaConf.to_object(
        OmegaConf.merge(
            OmegaConf.structured(Config), read, OmegaConf.from_dotlist(list(overrides))
        )
    )


def named_seed(seed: int, name: str) -> int:
    """Return the seed of one named draw, the same in every process and setup.

    Args:
        seed: The run's seed.
        name: What is drawn, such as `masks/CTX`.

    Returns:
        seed: That draw's own seed.
    """
    return zlib.crc32(f"{seed}/{name}".encode())
