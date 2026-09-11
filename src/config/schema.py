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
        split: The share of the features each split holds, in the order the
            code names the splits.
        seed: The number that fixes where a feature falls.
    """

    build: str
    root: str
    patchsize: dict[str, int] = field(default_factory=dict)
    split: list[float] = field(default_factory=list)
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
        patches: How many patches of each instrument one feature is drawn with.
        dim: How wide a token is in the instrument encoders and in the
            cross-sensor encoder, a multiple of 12 for the positional encoding.
        heads: How many attention heads those encoders run.
        depth: How many blocks each instrument encoder stacks.
        crossencoder_depth: How many blocks the cross-sensor encoder stacks.
        decoder_dim: How wide a token is in the decoders, a multiple of 12.
        decoder_heads: How many attention heads the decoders run.
        decoder_depth: How many blocks each decoder stacks.
    """

    name: str
    instruments: list[str]
    elevation: str
    patches: int
    dim: int
    heads: int
    depth: int
    crossencoder_depth: int
    decoder_dim: int
    decoder_heads: int
    decoder_depth: int


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
        mask_ratio: The share of each instrument's patches hidden from its
            encoder.
        temperature: What the contrastive term divides its similarities by.
        workers: How many processes read features beside the training.
        checkpoints: Where checkpoints are written, relative to the repository.
    """

    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_epochs: int
    patience: int
    mask_ratio: float
    temperature: float
    workers: int
    checkpoints: str


@dataclass
class EvaluationConfig:
    """What a trained model's latent space is measured over, and how.

    Attributes:
        split: The split the latents are read from, as the code names the
            splits.
        neighbours: How many nearest latents one retrieval reads.
        model: The published model to read, as it was published or as the key
            of one version of it, or None for the latest version of what a run
            of this architecture publishes.
    """

    split: str
    neighbours: int
    model: str | None = None


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
