# Multi-Sensor Masked Autoencoder for Mars

This repository trains a multimodal, multisensor, multiresolution masked autoencoder (MAE) over Mars remote sensing data. Given every source available for one tile of the Martian surface, the model learns a single latent representation of it, a foundation model for Mars remote sensing.

The samples it trains on come from [mars-multisensor-dataset](https://github.com/andreablushi/mars-multisensor-dataset), which splits Mars into equal-area tiles and crops each one's co-temporal CTX, CRISM, MOLA and SHARAD observations, drawn from NASA's [Orbital Data Explorer (ODE)](https://ode.rsl.wustl.edu/mars/). The built dataset is published on DigitalHub.

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
build = DatasetBuild(build_root(config.dataset.build, config.dataset.root))
```

A build published on DigitalHub is fetched an observation at a time, straight
from the store. Nothing fetched is written to that directory: a build runs to
some hundred gigabytes and a job's disk holds a fraction of it, so only a build
brought down whole is read off disk.

```python
from dhub.store import published_build

build = published_build(config.dataset.build, config.dataset.root)
```

Everything that reaches the platform is in `scripts/dhub`: the project the builds
are published in, the artifact each is published under, and the credentials a
long read mints again. `configs/dataset.yaml` names the build to read and where
builds sit, which a local read needs just as much.

## Evaluating what it learnt

The dataset repository builds a second dataset beside the training one: a
balanced draw of tiles, each labelled with the geological class the IAU
catalogue gives the feature it holds. It is published as its own build and
carries `labels.parquet` beside its index, which `src/evaluation/store.py`
reads with the tiles themselves.

```bash
uv run python scripts/evaluate.py             # the run named in configs/config.yaml
uv run python scripts/evaluate.py run_name=mae-deep
```

The model embeds every labelled tile into its grid of cells, and two tiles are
compared by a normalised Chamfer distance over those cells: each cell of one is
matched to the nearest cell of the other and the cost, a cosine distance
between two unit vectors, is averaged both ways.
`evaluation.minimal_chamfer_cell_distance` keeps a match near where the cell sits, so the arrangement of a feature counts
and not only what its places are made of; null matches a cell anywhere in the
other tile.

What comes out of that distance is retrieval at k of 1, 5, 10 and 20
(precision, recall and F1, a shared class its relevance), the silhouette
overall and per class, and the mean distance between each pair of classes with
its within, between and separation.

## Configs

`configs/` holds one file per section, `dataset.yaml`, `model.yaml`,
`training.yaml` and `evaluation.yaml`, composed by hydra into the run described
by `configs/config.yaml`, which also names the run with `run_name`:

```python
from config.load import load_config

config = load_config()  # the defaults
config = load_config(["dataset.patchsize.CTX=512"])  # one value of one
```

What comes back is `config.schema.Config`, read under the schema rather than
handed over as a bare mapping, so a key the schema does not declare is an error
rather than a line that settles nothing.
