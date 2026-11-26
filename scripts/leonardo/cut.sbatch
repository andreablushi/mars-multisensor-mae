#!/bin/bash
# Cuts every tile a config's stages read from $WORK/dataset into $FAST/ready, once per patch setup:
#   sbatch scripts/leonardo/cut.sbatch [key=value ...]
# submitted by submit.sh with the overrides of the runs that will read the cut.
#SBATCH --partition=lrd_all_serial
#SBATCH --cpus-per-task=4
#SBATCH --mem=30800M
#SBATCH --time=04:00:00
set -euo pipefail
"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/cut.py "$@" \
    dataset.store=disk "dataset.root=$WORK/dataset" "dataset.ready_root=$FAST/ready"
