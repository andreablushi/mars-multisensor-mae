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
- Overrides are dotted keys over `configs/`: `model.clutter_conditioning=false`,
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
  every score by pooling and view, the kNN confusion and class distance matrices,
  a crater tile and a north polar layered deposits tile rebuilt, and the vectors and
  reconstructions as the artifact `evaluation-<run name>`.
- A reconstruction shows each instrument's most measured observation, SHARAD's
  pass nearest 140° solar incidence, drawn over the configured clip or, on an
  open side, the original's own extreme.
- `random` is the baseline every score is coloured against.

## Comparison views

The W&B project holds two saved views. Training shows the curves of every training
run; Evaluation shows the score tables, the confusion and class distance matrices
and the reconstructions of every evaluation run, coloured against `random`. Its
tables and matrices are the W&B charts `scores` and `matrix` of the entity. Change
either view in the W&B page.

## Requirements for DigitalHub

A job installs `requirements.txt` when it starts. Export it again whenever
`uv.lock` changes, then commit and push it:

```bash
uv export --no-hashes --no-dev --group digitalhub --no-emit-project -o requirements.txt
```
