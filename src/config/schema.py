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
        split: The share of the tiles each split holds, in the code's order.
        seed: The number that fixes which split a tile falls in.
    """

    build: str
    root: str
    patchsize: dict[str, dict[str, int]]
    pool: dict[str, int]
    split: list[float]
    seed: int


@dataclass
class ModelConfig:
    """What the model reads, and how wide and deep each of its parts is.

    Attributes:
        instruments: The instruments whose patches become tokens, the delay one apart.
        delay: The instrument whose rows give every surface patch its delay.
        cell_m: How far a cell of a tile's grid runs along the ground, in metres.
        encoder_dim: How wide a token is everywhere but the decoders, a multiple of 12.
        encoder_heads: How many attention heads every encoder and the fusion run.
        encoder_depth: How many blocks each instrument encoder stacks.
        crossencoder_depth: How many blocks the cross-sensor encoder stacks.
        decoder_dim: How wide a token is in the decoders, a multiple of 12.
        decoder_heads: How many attention heads the decoders run.
        decoder_depth: How many blocks each decoder stacks.
    """

    instruments: list[str]
    delay: str
    cell_m: float
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
        max_steps: How many steps the run takes, at most.
        batch_size: How many tiles one step reads.
        learning_rate: The peak learning rate, reached after the warmup.
        weight_decay: The AdamW weight decay.
        warmup_steps: How many steps the rate climbs before the cosine decay.
        validate_every: How many steps between two validations.
        patience: How many validations without a lower loss before the run stops.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        checkpoints: Where checkpoints are written, relative to the repository.
    """

    max_steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    validate_every: int
    patience: int
    mask_ratio: float
    checkpoints: str


@dataclass
class EvaluationConfig:
    """What a trained model's latent space is measured over, and how.

    Attributes:
        build: The build the labelled tiles are read from, published as dataset-<build>.
        minimal_chamfer_cell_distance: How far, in cells along either axis, a
            cell may be matched from its own offset, or None to match it anywhere
            in the other tile.
    """

    build: str
    minimal_chamfer_cell_distance: int | None


@dataclass
class Config:
    """One run, composed of the build it reads, the model it trains, and how.

    Attributes:
        run_name: What the run is called, which names the model it saves and the
            tracked run it logs to.
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
