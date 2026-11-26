#!/bin/bash
# Evaluates one logged model offline on one A100 and 8 cores, a quarter of a Booster node:
#   sbatch scripts/leonardo/evaluate.sbatch <group>/<name> [--reconstruction] [--lavaflow] [key=value ...]
# submitted by submit.sh, which downloaded what it reads into the folder it writes in.
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
set -euo pipefail
export WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/evaluate.py "$@" \
    dataset.store=disk "dataset.root=$WORK/dataset" "dataset.ready_root=$FAST/ready"
