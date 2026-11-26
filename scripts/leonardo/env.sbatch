#!/bin/bash
# Builds the Leonardo env from the pins DigitalHub installs, the local clones editable on top:
#   sbatch scripts/leonardo/env.sbatch
# submitted by submit.sh, after each git pull that changes requirements.txt.
#SBATCH --partition=lrd_all_serial
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
export UV_PYTHON=3.13 UV_CACHE_DIR=$WORK/uv/cache UV_PYTHON_INSTALL_DIR=$WORK/uv/python
export VIRTUAL_ENV=$UV_PROJECT_ENVIRONMENT
rm -rf "$VIRTUAL_ENV"
uv venv "$VIRTUAL_ENV"
uv pip install -r requirements.txt
uv pip install --no-deps -e . -e ../mars-multisensor-dataset -e ../mars-multisensor-benchmark
