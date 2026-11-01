"""Reading the configs of one run under the schema, and the seeds it splits into."""

from __future__ import annotations

import zlib
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import fields

import torch
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
            OmegaConf.structured(Config),
            OmegaConf.merge(read, OmegaConf.from_dotlist(list(overrides))),
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


@contextmanager
def named_rng(seed: int, name: str) -> Iterator[None]:
    """Seed torch with one named draw, its state restored on the way out.

    Args:
        seed: The run's seed.
        name: What is drawn, such as `encoders.CTX`.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(named_seed(seed, name))
        yield


def named_generator(
    seed: int, name: str, device: torch.device | str = "cpu"
) -> torch.Generator:
    """Return a generator on one named draw of the seed.

    Args:
        seed: The run's seed.
        name: What is drawn, such as `split`.
        device: Where the generator draws.

    Returns:
        generator: The generator, seeded by that draw.
    """
    return torch.Generator(device=device).manual_seed(named_seed(seed, name))


def mask_generators(
    names: Iterable[str], seed: int, device: torch.device
) -> dict[str, torch.Generator]:
    """Return one mask generator per instrument, each on its own stream of the seed.

    Args:
        names: The instruments.
        seed: The run's seed.
        device: Where the masks are drawn.

    Returns:
        generators: Each instrument's generator, so its masks stay put whatever
            other instruments a setup reads.
    """
    return {name: named_generator(seed, f"masks/{name}", device) for name in names}
