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
```
