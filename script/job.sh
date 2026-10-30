#!/usr/bin/env bash
#SBATCH --account=research
#SBATCH --partition=gpuh
#SBATCH --qos=qos_research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

stage=${1:?stage required}
root=${2:?project root required}
round=${3:?round directory required}
previous=${4:-}
code=${5:?deployed source directory required}
max_tokens=${6:-512}
gpus=${7:?GPU count required}

test "$SLURM_JOB_NUM_NODES" -eq 1

bash "$code/script/worker.sh" prepare "$root" "$round" "$previous" "$code" "$max_tokens"
bash "$code/script/worker.sh" "$stage" "$root" "$round" "$previous" "$code" "$max_tokens" "$gpus"

if [ "$stage" = train ]; then
    touch "$round/training.ready"
fi
