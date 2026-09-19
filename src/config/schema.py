"""What a config may hold, which is what a composed one is read against."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """What a run reads of the published dataset, and how it cuts what it reads.

    Attributes:
        build: The build to read, which is published as dataset-<build>.
        root: Where every build sits on this machine, under a directory of its own name.
        patchsize: How far a patch runs along each cut axis, by instrument.
        split: The share of the tiles each split holds, in the code's order.
        overlap: The share of each other sensor's patches over the anchor's ground.
        seed: The number that fixes where a tile falls.
    """

    build: str
    root: str
    patchsize: dict[str, int]
    split: list[float]
    overlap: float
    seed: int


@dataclass
class ModelConfig:
    """What the model reads, and how wide and deep each of its parts is.

    Attributes:
        name: The architecture.
        instruments: The instruments whose patches become tokens, the elevation apart.
        elevation: The instrument whose values give every surface patch its height.
        cell_m: How far a cell of a tile's grid runs along the ground, in metres.
        encoder_dim: How wide a token is everywhere but the decoders, a multiple of 12.
        encoder_heads: How many attention heads every encoder and the fusion run.
        encoder_depth: How many blocks each instrument encoder stacks.
        crossencoder_depth: How many blocks the cross-sensor encoder stacks.
        decoder_dim: How wide a token is in the decoders, a multiple of 12.
        decoder_heads: How many attention heads the decoders run.
        decoder_depth: How many blocks each decoder stacks.
    """

    name: str
    instruments: list[str]
    elevation: str
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
        epochs: How many passes over the training tiles, at most.
        batch_size: How many tiles one step reads, which settles patches per read.
        patches_per_step: How many patches one step carries, set by the widest.
        learning_rate: The peak learning rate, reached after the warmup.
        weight_decay: The AdamW weight decay.
        warmup_epochs: How many epochs the rate climbs before the cosine decay.
        patience: How many epochs without a lower validation loss before the run stops.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        consistency: What a grid of one instrument agreeing with the whole counts.
        uniformity: What the cells standing apart from each other counts.
        checkpoints: Where checkpoints are written, relative to the repository.
    """

    epochs: int
    batch_size: int
    patches_per_step: int
    learning_rate: float
    weight_decay: float
    warmup_epochs: int
    patience: int
    mask_ratio: float
    consistency: float
    uniformity: float
    checkpoints: str


@dataclass
class EvaluationConfig:
    """What a trained model's latent space is measured over, and how.

    Attributes:
        split: The split the latents are read from, as the code names the splits.
        neighbours: How many nearest latents one retrieval reads.
        model: The published model to read, or None for the latest of this one.
    """

    split: str
    neighbours: int
    model: str | None


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
