#!/bin/bash
# Syncs one job's offline runs to W&B on the free serial partition, which has internet:
#   sbatch scripts/leonardo/sync.sbatch <job folder>
# submitted by submit.sh to start once the GPU job succeeds.
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
set -euo pipefail
"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/leonardo/sync.py "$1"
