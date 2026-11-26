#!/bin/bash
# Submits one stage from a Leonardo login node, env, cut, train or evaluate:
#   scripts/leonardo/submit.sh env
#   scripts/leonardo/submit.sh cut [key=value ...]
#   scripts/leonardo/submit.sh <train|evaluate> <group>/<name> [flags] [key=value ...]
# Downloads here what an evaluation reads from W&B, queues the offline GPU job, then
# the sync job on lrd_all_serial once it succeeds. Everything the jobs write sits in
# one folder under $WORK/jobs, removed by the sync; SLURM logs go to $WORK/logs.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a
source .env
set +a
stage=$1
test -f "scripts/leonardo/$stage.sbatch"
mkdir -p "$WORK/jobs" "$WORK/logs"
export UV_PROJECT_ENVIRONMENT=$WORK/envs/mars-multisensor-mae
logs=$WORK/logs/%x-%j.out
if [[ $stage == env || $stage == cut ]]; then
    sbatch --job-name="$stage" --output="$logs" "scripts/leonardo/$stage.sbatch" "${@:2}"
    exit
fi
run_name=$2
flat=${run_name//\//.}
MAE_OUTPUT_ROOT=$(mktemp -d "$WORK/jobs/$flat.XXXXXX")
export MAE_OUTPUT_ROOT
export WANDB_DIR=$MAE_OUTPUT_ROOT
export WANDB_DATA_DIR=$MAE_OUTPUT_ROOT/wandb-data
export WANDB_CACHE_DIR=$MAE_OUTPUT_ROOT/wandb-cache
trap 'rm -rf "$MAE_OUTPUT_ROOT"' ERR
if [[ $stage == evaluate ]]; then
    flags=()
    for argument in "${@:3}"; do
        if [[ $argument == --* ]]; then
            flags+=("$argument")
        fi
    done
    "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/leonardo/prepare.py "$run_name" "${flags[@]}"
fi
job=$(sbatch --parsable --job-name="$stage-$flat" --output="$logs" "scripts/leonardo/$stage.sbatch" "${@:2}")
trap - ERR
sync=$(sbatch --parsable --job-name="sync-$flat" --output="$logs" --dependency="afterok:$job" --kill-on-invalid-dep=yes scripts/leonardo/sync.sbatch "$MAE_OUTPUT_ROOT")
echo "$stage $job, sync $sync, folder $MAE_OUTPUT_ROOT"
