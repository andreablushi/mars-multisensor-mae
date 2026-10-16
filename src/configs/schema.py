"""What a config may hold, which is what a composed one is read against."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """What a run reads of the published dataset, and how it cuts what it reads.

    Attributes:
        build: The build to read, which is published as dataset-<build>.
        root: Where every build sits on this machine, under a directory of its own name.
        patchsize: How far a patch runs along an axis holding each thing, by
            instrument, an axis it omits taken whole.
        pool: How many ground samples of a patch are averaged into one, by
            instrument, one left out read whole.
        budget: How many patches of each instrument a training or validation read
            keeps, the ones nearest a centre drawn on the tile.
        budget_scale: The most a training batch's budget is multiplied by, drawn
            from 1 for each batch.
        split: The share of the observations each split holds, in the code's order.
        seed: The number that fixes which split a tile falls in.
    """

    build: str
    root: str
    patchsize: dict[str, dict[str, int]]
    pool: dict[str, int]
    budget: dict[str, int]
    budget_scale: float
    split: list[float]
    seed: int


@dataclass
class ModelConfig:
    """What the model reads, and how wide and deep each of its parts is.

    Attributes:
        instruments: The instruments whose patches become tokens.
        encoder_dim: How wide a token is everywhere but the decoders, a multiple of 6.
        encoder_heads: How many attention heads every encoder runs.
        encoder_depth: How many blocks each instrument encoder stacks.
        crossencoder_depth: How many blocks the cross-sensor encoder stacks.
        decoder_dim: How wide a token is in the decoders, a multiple of 6.
        decoder_heads: How many attention heads the decoders run.
        decoder_depth: How many blocks each decoder stacks.
    """

    instruments: list[str]
    encoder_dim: int
    encoder_heads: int
    encoder_depth: int
    crossencoder_depth: int
    decoder_dim: int
    decoder_heads: int
    decoder_depth: int


@dataclass
class TrainingConfig:
    """How a run trains, validates and stops.

    Attributes:
        epochs: How many passes over the training split the run takes.
        batch_size: How many tiles one step's gradient is averaged over.
        learning_rate: The peak learning rate, reached after the warmup.
        weight_decay: The AdamW weight decay.
        warmup_epochs: How many epochs the rate climbs before the cosine decay.
        validate_every: How many epochs between two validations.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        cmr: The instruments each instrument is rebuilt from cross-modally.
        checkpoints: Where checkpoints are written, relative to the repository.
    """

    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_epochs: int
    validate_every: int
    mask_ratio: float
    cmr: dict[str, list[str]]
    checkpoints: str


@dataclass
class EvaluationConfig:
    """What a trained model's latent space is measured over, and how.

    Attributes:
        build: The build the labelled tiles are read from, published as dataset-<build>.
    """

    build: str


@dataclass
class Config:
    """One run, composed of the build it reads, the model it trains, and how.

    Attributes:
        run_name: What the run is called, which names its tracked run and its model,
            the part before a slash grouping it.
        dataset: What it reads.
        model: What it trains.
        training: How it trains.
        evaluation: How what it trained is measured.
    """

    run_name: str
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
