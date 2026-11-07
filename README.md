# Multi-Sensor Masked Autoencoder for Mars

A multimodal, multisensor masked autoencoder for Mars remote sensing, trained on
the tiles of [mars-multisensor-dataset](https://github.com/andreablushi/mars-multisensor-dataset).

## Setup

```bash
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
- `--evaluate` evaluates the model once trained.
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
  every score by pooling and view, the class distance matrices, the tiles
  `evaluation.tiles` names rebuilt, and the vectors and reconstructions as the
  artifact `evaluation-<run name>`.
- A reconstruction shows the observation `evaluation.tiles` names for each
  instrument: the original, the input with its hidden patches in blue, UMR with
  the visible patches kept, and CMR from every other instrument read whole. CRISM
  gets two rows, its 770 nm albedo and a false colour composite. Values are drawn
  over the configured clip or, on an open side, the original's own extreme.
- Below them is where CMR looked: the decoder's attention rolled out over its
  blocks (Abnar and Zuidema, 2020), its heads averaged, red where a place drew
  more than its instrument's mean and blue where less, each instrument's share in
  the caption. SHARAD shows its most attended pass. The rebuilt observation is
  outlined in orange on every map, a SHARAD pass as its track.
- MOLA and SHARAD are also rebuilt from each other alone.
- `random` is the baseline every score is coloured against.

## Benchmark

```bash
uv run python scripts/benchmark.py default/mae-small
uv run --group digitalhub python scripts/benchmark.py --dh --ref main default/mae-small
```

- It probes and fine-tunes two heads on SWIM's subsurface ice consistency over the
  run's latest W&B model, and logs a benchmark run of the same name.
- Both heads regress SWIM's consistency, the tile head the tile's median and the
  map head every grid point, trained together on one read of the tokens. A
  training crop scores only the points it has tokens near. Fine-tuning trains LoRA
  updates of the encoders.
- RMSE and R2 measure the fit, the map head's as each point's departure from its
  tile's mean, over the tiles whose consistency varies. Each head is kept at its
  best validation RMSE.
- The drawn tiles show SWIM's consistency beside each map head's, red where icy and
  blue where dry, a row per depth.
- Fewer `benchmark.instruments` than the model reads name the run for those left
  out, such as `default/mae-small-no-sharad`.
- `--evaluate` on `scripts/train.py` also runs the benchmark.

## Comparison views

The W&B project holds two saved views. Training shows the curves of every training
run; Evaluation shows the score tables, the class distance matrices and the
reconstructions of every evaluation run, coloured against `random`. Its
tables and matrices are the W&B charts `scores` and `matrix` of the entity. Change
either view in the W&B page.

## Requirements for DigitalHub

A job installs `requirements.txt` when it starts. Export it again whenever
`uv.lock` changes, then commit and push it:

```bash
uv export --no-hashes --no-dev --group digitalhub --no-emit-project -o requirements.txt
```
