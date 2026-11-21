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
- `--evaluate` evaluates the model once trained.
- `configs/digitalhub.yaml` sets the project, profile, cores, memory and disk,
  overridden alike: `digitalhub.stages.evaluation.profile=1xrtxa5000-shared`.
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

The SWIM ice benchmark of a logged model lives in
[mars-multisensor-finetune](https://github.com/andreablushi/mars-multisensor-finetune).

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
