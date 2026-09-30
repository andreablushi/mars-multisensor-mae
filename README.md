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
uv run python scripts/train.py                                  # here
uv run python scripts/train.py run_name=mae-deep                # with hydra overrides
uv run --group digitalhub python scripts/train.py --dh --ref floor --evaluate run_name=mae-deep
```

- `--dh` submits to DigitalHub, `--ref` names the pushed branch or tag it runs.
- `--evaluate` evaluates the model once trained.
- `configs/digitalhub.yaml` sets the project, profile, cores, memory and disk.

## Evaluation

```bash
uv run python scripts/evaluate.py                               # the run in configs/config.yaml
uv run python scripts/evaluate.py run_name=mae-deep
uv run --group digitalhub python scripts/evaluate.py --dh --ref floor run_name=mae-deep
```

## Requirements for DigitalHub

A job installs `requirements.txt` when it starts. Export it again whenever
`uv.lock` changes, then commit and push it:

```bash
uv export --no-hashes --no-dev --group digitalhub --no-emit-project -o requirements.txt
```
