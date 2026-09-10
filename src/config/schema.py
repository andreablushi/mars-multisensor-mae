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
    """What the model reads, and how wide and deep each of its parts is.

    Attributes:
        name: The architecture.
        instruments: The instruments whose patches become tokens, as ODE names
            them. The elevation instrument is not one of them.
        elevation: The instrument whose values give every surface patch its
            height.
        bands: How many bands a spectral patch is resampled to along its
            wavelength axis, so every one of them is the same shape.
        patches: How many patches of each instrument one feature is drawn with.
        dim: How wide a token is in the instrument encoders and in the
            cross-sensor encoder, a multiple of 6 for the position encoding.
        heads: How many attention heads those encoders run.
        depth: How many blocks each instrument encoder stacks.
        fusion_depth: How many blocks the cross-sensor encoder stacks.
        latent: How many dimensions the sphere every token is placed on has.
        kappa: How tightly a sampled token stays about its mean direction.
        decoder_dim: How wide a token is in the decoders, a multiple of 6.
        decoder_heads: How many attention heads the decoders run.
        decoder_depth: How many blocks each decoder stacks.
        mask_ratio: The share of each instrument's patches hidden from its
            encoder.
    """

    name: str
    instruments: list[str]
    elevation: str
    bands: int
    patches: int
    dim: int
    heads: int
    depth: int
    fusion_depth: int
    latent: int
    kappa: float
    decoder_dim: int
    decoder_heads: int
    decoder_depth: int
    mask_ratio: float


@dataclass
class TrainingConfig:
    """How a run trains, validates and stops.

    Attributes:
        epochs: How many passes over the training features, at most.
        batch_size: How many features one step reads.
        learning_rate: The peak learning rate, reached after the warmup.
        weight_decay: The AdamW weight decay.
        warmup_epochs: How many epochs the learning rate climbs over before
            the cosine decay.
        patience: How many epochs without a lower validation loss before the
            run stops.
        uniformity_weight: How much the batch uniformity term weighs against
            the reconstruction.
        neighbours: How many nearest features a retrieval metric looks at.
        workers: How many processes read features beside the training.
        checkpoints: Where checkpoints are written, relative to the repository.
        project: The Weights & Biases project the run is tracked under.
    """

    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_epochs: int
    patience: int
    uniformity_weight: float
    neighbours: int
    workers: int
    checkpoints: str
    project: str


@dataclass
class Config:
    """One run, composed of the build it reads, the model it trains, and how.

    Attributes:
        dataset: What it reads.
        model: What it trains.
        training: How it trains.
    """

    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
