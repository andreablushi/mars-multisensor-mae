# Multi-Sensor Masked Autoencoder for Mars

A multimodal, multisensor masked autoencoder for Mars remote sensing, trained on
the tiles of [mars-multisensor-dataset](https://github.com/andreablushi/mars-multisensor-dataset).

## Setup

The dataset installs editable from a clone beside this one, so local code always
runs the dataset as checked out there.

```bash
git clone https://github.com/andreablushi/mars-multisensor-dataset ../mars-multisensor-dataset
uv sync
source .venv/bin/activate
git config core.hooksPath .githooks            # once per clone, ruff runs before a push
uv run ruff check . && uv run ruff format .    # lint, over the whole repo
```

## DigitalHub setup

```bash
uv sync --group digitalhub
dhcli register <your-digitalhub-core-endpoint>
dhcli login                                    # opens a browser tab
cp .env.example .env                           # once, then fill it in
```

The platform project needs these secrets: `DHCORE_PERSONAL_ACCESS_TOKEN`,
`WANDB_ENTITY`, `WANDB_PROJECT` and `WANDB_API_KEY`.

## Training

```bash
uv run python scripts/train.py default/mae-small                          # here
uv run python scripts/train.py default/mae-small training.batch_size=64   # with overrides
uv run --group digitalhub python scripts/train.py --dh --ref main --evaluate default/mae-small
```

- The run name is required: it names the W&B run, its model and its evaluation,
  the part before a slash grouping it.
- Overrides are dotted keys over `configs/`: `training.acquisition=false`,
  `training.contrastive.instruments=[CTX,CRISM]`.
- `--dh` submits to DigitalHub, `--ref` names the pushed branch or tag it runs.
- `--evaluate` evaluates the model once trained, `--benchmark` benchmarks it.
- `configs/digitalhub.yaml` sets the project, profile, cores, memory and disk.
- The last checkpoint is logged to W&B as the model `model-<run name>`, slashes
  made dots.

## Evaluation

```bash
uv run python scripts/evaluate.py default/mae-small
uv run python scripts/evaluate.py random                                # untrained weights
uv run --group digitalhub python scripts/evaluate.py --dh --ref main default/mae-small
```

- It reads the run's latest W&B model and logs an evaluation run of the same name:
  every score by pooling and view, cross-sensor retrieval among them (one instrument's
  vector asked against another's: the mAP of the tile's own vector, one over its
  rank, per query and gallery), the class distance matrices, the tiles
  `evaluation.tiles` names rebuilt, and the vectors and reconstructions as the
  artifact `evaluation-<run name>`.
- A reconstruction shows the observation `evaluation.tiles` names for each
  instrument: the original, the input with its hidden patches in blue, UMR with
  the visible patches kept, and CMR from every other instrument read whole. CRISM
  gets two rows, its 770 nm albedo and a false colour composite. Values are drawn
  over the configured clip or, on an open side, the original's own extreme.
- Below them is where CMR looked: the decoder's attention rolled out over its
  blocks (Abnar and Zuidema, 2020), its heads averaged, each place coloured by what
  it drew from the rebuilt patches within 3 km of it on the ground: red above a
  uniform share over every token and blue below, on one scale for every
  instrument, grey where no rebuilt patch lies near. Each instrument's share is in
  the caption. SHARAD shows its most attended pass. The rebuilt observation is
  outlined in orange on every map, a SHARAD pass as its track.
- MOLA and SHARAD are also rebuilt from each other alone.
- `random` is the baseline every score is coloured against.

## Benchmark

```bash
uv run python scripts/benchmark.py default/mae-small
uv run --group digitalhub python scripts/benchmark.py --dh --ref main default/mae-small
```

- It trains an attentive probe on SWIM's subsurface ice consistency, 1 to 5 m and
  below 5 m, and on its radar surface power, which comes from SHARAD alone, over
  the run's latest W&B model, frozen and fine-tuned, both in one loop on the same
  batches and from the same draw, once on each share of the training tiles
  `benchmark.fractions` names (1%, 5%, 10% and all of them), a smaller share held
  in every larger one. It logs two runs of the same name:
  `benchmark-finetune` holds the training curves and the heads, `benchmark` the
  test scores.
- `--finetune` fine-tunes alone, `--evaluate` scores the last logged heads alone, and
  neither runs both.
- The probe regresses SWIM's three targets at every grid point of a tile, each tile
  read whole in training as in test. Fine-tuning trains the encoders and the
  cross-sensor encoder whole, so on `random` it is training from scratch. The loss
  divides each target by its spread over the training tiles.
- RMSE and R2 measure the fit over every SWIM cell of the test tiles. Each protocol
  is kept at its best validation R2, averaged over the targets.
- The drawn tiles show SWIM's targets beside the head's, a row per target, each
  coloured in viridis over the range its panels span.
- Fewer `benchmark.instruments` than the model reads name the run for those left
  out, such as `default/mae-small-no-sharad`.
- `--benchmark` on `scripts/train.py` also runs the benchmark.

## Comparison views

The W&B project holds two saved views. Training shows the curves of every training
run; Evaluation shows the score tables, the class distance matrices and the
reconstructions of every evaluation run, coloured against `random`. Its
tables and matrices are the W&B charts `scores` and `matrix` of the entity. Change
either view in the W&B page.

## Requirements for DigitalHub

A job installs `requirements.txt` when it starts, which takes the dataset from its
GitHub main. Export it again whenever `uv.lock` changes, then commit and push it:

```bash
uv export --no-hashes --no-dev --group digitalhub --no-emit-project --no-sources -o requirements.txt
uv lock                                        # the export relocks without sources
```
