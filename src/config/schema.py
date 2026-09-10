"""What a config may hold, which is what a composed one is read against."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    """What a run reads of the published dataset, and how it cuts what it reads.

    Attributes:
        build: The build to read, which is published as dataset-<build>.
        root: Where every build sits on this machine, under a directory of its
            own name.
        patchsize: How far a patch runs along every axis it is cut on, by
            instrument, and under "default" for every instrument unnamed.
        split: The share of the features each split holds, by its name.
        seed: The number that fixes where a feature falls.
    """

    build: str
    root: str
    patchsize: dict[str, int] = field(default_factory=dict)
    split: dict[str, float] = field(default_factory=dict)
    seed: int = 42


@dataclass
class ModelConfig:
    """What a run trains.

    Attributes:
        name: The architecture, which is all it says until there is an encoder.
    """

    name: str


@dataclass
class Config:
    """One run, composed of the build it reads and the model it trains.

    Attributes:
        dataset: What it reads.
        model: What it trains.
    """

    dataset: DatasetConfig
    model: ModelConfig
