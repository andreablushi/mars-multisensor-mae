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
```

A read outlasts the credentials it is started with: those lapse after some six
hours, and a training run still going then can fetch nothing. So it mints its
own instead. It presents a personal access token, which the platform holds as a
secret named `DHCORE_PERSONAL_ACCESS_TOKEN` and hands the job under that name.
The token names neither who issues credentials nor who asks for them, so `.env`
carries those two, the same pair the dataset repository's own `.env` holds.

## Reading the dataset

`src/dataset` reads one build out of one directory, `data/dataset/<build>`, and
knows nothing of where that build came from. A build brought down whole sits
there and is read off disk:

```python
from dataset.config import build_root, load_config
from dataset.store import DatasetBuild

config = load_config()
build = DatasetBuild(build_root(config))
```

A build published on DigitalHub is fetched a crop at a time into that same
directory, so a later pass over the same crops asks the platform for none of
them:

```python
from dh.store import published_build

build = published_build(config)
```

Everything that reaches the platform is in `scripts/dh`: the project the builds
are published in, the artifact each is published under, and the credentials a
long read mints again. `src/dataset/config.yaml` names the build to read and
where builds sit, which a local read needs just as much.
