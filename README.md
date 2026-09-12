# Multi-Sensor Masked Autoencoder for Mars

This repository trains a multimodal, multisensor, multiresolution masked autoencoder (MAE) over Mars remote sensing data. Given every source available for a geological feature, the model learns a single latent representation of it, a foundation model for Mars remote sensing.

The samples it trains on come from [mars-multisensor-dataset](https://github.com/andreablushi/mars-multisensor-dataset), which selects geological features from NASA's [Orbital Data Explorer (ODE)](https://ode.rsl.wustl.edu/mars/) and crops each one's co-temporal CTX, CRISM and SHARAD observations. The built dataset is published on DigitalHub.

## Development commands

```bash
uv sync                                     # environment
source .venv/bin/activate                   # activate it, at the start of a session
uv run ruff check . && uv run ruff format . # lint, over the whole repo
```

A pre-push hook blocks a push when ruff is unhappy. Enable it once per clone:

```bash
git config core.hooksPath .githooks
```

## Running it on DigitalHub

The dataset lives on [DigitalHub](https://scc-digitalhub.github.io/docs/0.15/), which also runs the training on a cluster and keeps what it produced as versioned entities.

```bash
uv sync --group digitalhub
dhcli register <your-digitalhub-core-endpoint>
dhcli login                                    # opens a browser tab

cp .env.example .env                           # once, then fill it in

uv run --group digitalhub python scripts/train.py --dh --ref refine
```

Everything a submission needs, from the project name to the box a job asks
for, is in `configs/digitalhub.yaml`. The pip requirements are taken straight
from `pyproject.toml`, so the image always matches this repository. The
platform clones a pushed commit, so every change has to be on the branch
`--ref` names before it can run.

A read outlasts the credentials it is started with: those lapse after some six
hours, and a training run still going then can fetch nothing. So it mints its
own instead. It presents a personal access token, which the platform holds as a
secret named `DHCORE_PERSONAL_ACCESS_TOKEN` and hands the job under that name.
The token names neither who issues credentials nor who asks for them, so `.env`
carries those two, the same pair the dataset repository's own `.env` holds. It
also carries the Weights & Biases entity, project and key a run here is tracked
with; a job is handed the same three as platform secrets of the same names.

## Reading the dataset

`src/dataset` reads one build out of one directory, `data/dataset/<build>`, and
knows nothing of where that build came from. A build brought down whole sits
there and is read off disk:

```python
from config.load import load_config
from config.paths import build_root
from dataset.store import DatasetBuild

config = load_config()
build = DatasetBuild(build_root(config.dataset))
```

A build published on DigitalHub is fetched an observation at a time, straight
from the store. Nothing fetched is written to that directory: a build runs to
some hundred gigabytes and a job's disk holds a fraction of it, so only a build
brought down whole is read off disk.

```python
from dh.store import published_build

build = published_build(config.dataset)
```

Everything that reaches the platform is in `scripts/dh`: the project the builds
are published in, the artifact each is published under, and the credentials a
long read mints again. `configs/dataset/<build>.yaml` names the build to read and
where builds sit, which a local read needs just as much.

## Configs

`configs/` holds one file per build under `dataset/` and one per architecture
under `model/`, composed by hydra into the run described by `configs/config.yaml`:

```python
from config.load import load_config

config = load_config()  # the defaults
config = load_config(["dataset=small"])  # another file of a group
config = load_config(["dataset.seed=7"])  # one value of one
```

What comes back is `config.schema.Config`, read under the schema rather than
handed over as a bare mapping, so a key the schema does not declare is an error
rather than a line that settles nothing.
