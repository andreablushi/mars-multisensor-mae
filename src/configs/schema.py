"""What a config may hold, which is what a composed one is read against."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """What a run reads of the published dataset, and how it cuts what it reads.

    Attributes:
        build: The build to read, which is published as dataset-<build>.
        root: Where every build sits on this machine, under a directory of its own name.
        clip: The low and high each instrument's values are clipped to, None for open.
        sharad_atmosphere_cropping: Whether SHARAD is cropped below its atmosphere.
        acquisition: The acquisition angles of each instrument, as the index names them.
        patchsize: A patch's extent per axis and instrument, an omitted axis whole.
        budget: How many patches of each instrument a training or validation read keeps.
        budget_scale: The most a training batch's budget is multiplied by.
        split: The share of the tiles the training and the validation hold.
        seed: The number that fixes which split a tile falls in.
    """

    build: str
    root: str
    clip: dict[str, list[float | None]]
    sharad_atmosphere_cropping: bool
    acquisition: dict[str, list[str]]
    patchsize: dict[str, dict[str, int]]
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
class ContrastiveConfig:
    """Whether CSMAE's MIM aligns instruments across tiles, and over what.

    Attributes:
        pooling: What each instrument's tile vector is, its GAP or its CLS.
        instruments: The instruments aligned, each pair its own term, none for no MIM.
        temperature: What every cosine is divided by.
    """

    pooling: str
    instruments: list[str]
    temperature: float


@dataclass
class TrainingConfig:
    """How a run trains, validates and stops.

    Attributes:
        num_steps: How many steps the run takes.
        batch_size: How many tiles one step's gradient is averaged over.
        learning_rate: The peak learning rate, reached after the warmup.
        weight_decay: The AdamW weight decay.
        warmup_steps: How many steps the rate climbs before the cosine decay.
        validate_every: How many steps between two validations.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        reconstruction_error: What UMR and CMR measure, MSE or RMSE.
        contrastive: Whether MIM aligns instruments across tiles, and over what.
        acquisition: Whether the encoders and decoders read the acquisition angles.
        checkpoints: Where checkpoints are written, relative to the repository.
    """

    num_steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    validate_every: int
    mask_ratio: float
    reconstruction_error: str
    contrastive: ContrastiveConfig
    acquisition: bool
    checkpoints: str


@dataclass
class EvaluationConfig:
    """What a trained model's latent space is measured over, and how.

    Attributes:
        build: The build the labelled tiles are read from, published as dataset-<build>.
        tiles: The tiles drawn as reconstructions and attention maps, by what they show.
        ranging: The two ranging instruments, each also rebuilt from the other alone.
    """

    build: str
    tiles: dict[str, str]
    ranging: list[str]


@dataclass
class Config:
    """One run, composed of the build it reads, the model it trains, and how.

    Attributes:
        dataset: What it reads.
        model: What it trains.
        training: How it trains.
        evaluation: How what it trained is measured.
    """

    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
