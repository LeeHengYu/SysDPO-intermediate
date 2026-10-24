#!/usr/bin/env bash
#SBATCH --account=research
#SBATCH --partition=gpuh
#SBATCH --qos=qos_research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

stage=${1:?stage required}
root=${2:?project root required}
round=${3:?round directory required}
previous=${4:-}
code=${5:?deployed source directory required}
max_tokens=${6:-512}

test "$SLURM_JOB_NUM_NODES" -eq 1

# One launcher sees all allocated GPUs; torchrun assigns one GPU to each worker.
srun --kill-on-bad-exit=1 --ntasks=1 --gpus-per-task="h100:$SLURM_NTASKS" --gpu-bind=none \
    bash "$code/script/worker.sh" prepare "$root" "$round" "$previous" "$code" "$max_tokens"

srun --kill-on-bad-exit=1 --ntasks=1 --gpus-per-task="h100:$SLURM_NTASKS" \
    --gpu-bind=none \
    bash "$code/script/worker.sh" "$stage" "$root" "$round" "$previous" "$code" "$max_tokens" "$SLURM_NTASKS"

if [ "$stage" = train ]; then
    touch "$round/training.ready"
fi
