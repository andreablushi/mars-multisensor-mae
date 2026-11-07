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
        reconstruction_error: What UMR and CMR measure, MSE, RMSE or RMSE+GRADIENT.
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
class DrawnTile:
    """One tile drawn as reconstructions and attention maps.

    Attributes:
        tile: The tile.
        shown: The observation of each instrument shown, by its identifier.
    """

    tile: str
    shown: dict[str, str]


@dataclass
class EvaluationConfig:
    """What a trained model's latent space is measured over, and how.

    Attributes:
        build: The build the labelled tiles are read from, published as dataset-<build>.
        batch_size: How many tiles one batch of the vector pass holds, each read whole.
        neighbours: How many neighbours the kNN vote reads.
        tiles: The tiles drawn as reconstructions and attention maps, by what they show.
        ranging: The two ranging instruments, each also rebuilt from the other alone.
    """

    build: str
    batch_size: int
    neighbours: int
    tiles: dict[str, DrawnTile]
    ranging: list[str]


@dataclass
class FinetuneConfig:
    """How every benchmark head is trained, as a linear probe and fine-tuned.

    Attributes:
        batch_size: How many tiles one batch holds.
        budget: How many patches of each instrument a tile keeps.
        budget_scale: The most a batch's budget is multiplied by.
        epochs: The most epochs a head trains for.
        patience: How many epochs without a better validation AUPRC stop it.
        warmup_epochs: How many epochs the rate climbs before the cosine decay.
        head_learning_rate: The head's peak learning rate.
        encoder_learning_rate: The encoders' peak learning rate when fine-tuned.
        weight_decay: The AdamW weight decay.
        max_grad_norm: The norm every step's gradient is clipped to.
    """

    batch_size: int
    budget: dict[str, int]
    budget_scale: float
    epochs: int
    patience: int
    warmup_epochs: int
    head_learning_rate: float
    encoder_learning_rate: float
    weight_decay: float
    max_grad_norm: float


@dataclass
class BenchmarkEvaluationConfig:
    """How every benchmark head is measured on the validation and test tiles.

    Attributes:
        batch_size: How many tiles one batch holds, each read whole.
        threshold: The chance at or above which a point or tile counts as icy, for
            mIoU, the confusion and the drawn grids.
        tiles: The test tiles drawn as SWIM's grid beside the predicted ones, by
            what they show.
        shown: The instrument whose every observation of a tile is drawn beneath.
    """

    batch_size: int
    threshold: float
    tiles: dict[str, str]
    shown: str


@dataclass
class BenchmarkConfig:
    """What a trained model is probed and fine-tuned on: SWIM's ice under each tile.

    Attributes:
        build: The build whose labelled tiles are read, published as dataset-<build>.
        instruments: The instruments the heads read.
        folds: How many equal folds the labelled tiles are dealt into, the first
            testing, the second validating and the rest training.
        block_deg: How wide a square of the map is, in degrees, each one dealt whole.
        pooling: What the classifier reads of a tile, the GAP of its tokens or its CLS.
        segmenter_depth: How many cross-attention blocks the segmenter stacks.
        finetune: How every head is trained.
        evaluation: How every head is measured.
    """

    build: str
    instruments: list[str]
    folds: int
    block_deg: float
    pooling: str
    segmenter_depth: int
    finetune: FinetuneConfig
    evaluation: BenchmarkEvaluationConfig


@dataclass
class Config:
    """One run, composed of the build it reads, the model it trains, and how.

    Attributes:
        dataset: What it reads.
        model: What it trains.
        training: How it trains.
        evaluation: How what it trained is measured.
        benchmark: How what it trained is probed and fine-tuned on SWIM's ice.
    """

    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    benchmark: BenchmarkConfig
