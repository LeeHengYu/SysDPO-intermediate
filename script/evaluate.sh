#!/usr/bin/env bash
#SBATCH --account=research
#SBATCH --partition=gpuh
#SBATCH --qos=qos_research
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail
root=${1:?root required}
output=${2:?evaluation directory required}
checkpoint=${3:?checkpoint required}
code=${4:?release required}
scratch="/mnt/slurm_scratch/job_${SLURM_JOB_ID:?}"
test -d "$scratch"
test "$SLURM_JOB_NUM_NODES" -eq 1

mkdir -p "$scratch/base" "$scratch/checkpoint"
rsync -a "$root/models/Qwen1.5-1.8B-Chat/" "$scratch/base/"
for name in A_adapter B_adapter; do
    rsync -a "$checkpoint/$name" "$scratch/checkpoint/"
done
cp "$checkpoint/complete" "$scratch/checkpoint/complete"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export HF_HOME="$scratch/huggingface" TMPDIR="$scratch"
srun --kill-on-bad-exit=1 --ntasks=1 --gpus-per-task="h100:$SLURM_NTASKS" --gpu-bind=none \
    "$code/.venv/bin/python" -u -m torch.distributed.run --standalone \
    --nproc-per-node="$SLURM_NTASKS" -m multi_llm.evaluate generate \
    --base "$scratch/base" --checkpoint "$scratch/checkpoint" --directory "$output"
