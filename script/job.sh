#!/usr/bin/env bash
#SBATCH --account=research
#SBATCH --partition=gpuh
#SBATCH --qos=qos_research
#SBATCH --nodes=1-2
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
max_tokens=${6:-1024}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))

# Stage once per node; local scratch is not shared between nodes.
srun --kill-on-bad-exit=1 --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
    bash "$code/script/worker.sh" prepare "$root" "$round" "$previous" "$code" "$max_tokens"

srun --kill-on-bad-exit=1 --ntasks="$SLURM_NTASKS" --gpu-bind=none \
    bash "$code/script/worker.sh" "$stage" "$root" "$round" "$previous" "$code" "$max_tokens"

if [ "$stage" = train ]; then
    touch "$round/training.ready"
fi
