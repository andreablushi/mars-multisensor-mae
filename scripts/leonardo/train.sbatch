#!/bin/bash
# Trains one run offline on one A100 and 8 cores, a quarter of a Booster node:
#   sbatch scripts/leonardo/train.sbatch <group>/<name> [--evaluate] [key=value ...]
# submitted by submit.sh, which sets the folder it writes in. The checkpoint is written
# where an offline evaluation finds the model's artifact.
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=48:00:00
set -euo pipefail
export WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/train.py "$@" \
    dataset.store=disk "dataset.root=$WORK/dataset" "dataset.ready_root=$FAST/ready" \
    "training.checkpoints=$MAE_OUTPUT_ROOT/artifacts/model-${1//\//.}"
