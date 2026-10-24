#!/usr/bin/env bash
set -euo pipefail

stage=${1:?stage required}
root=${2:?project root required}
round=${3:?round directory required}
previous=${4:-}
code=${5:?deployed source directory required}
max_tokens=${6:-512}
workers=${7:-1}
scratch="/mnt/slurm_scratch/job_${SLURM_JOB_ID:?Slurm job required}"
test -d "$scratch"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export HF_HOME="$scratch/huggingface" TMPDIR="$scratch"

python="$code/.venv/bin/python"

if [ "$stage" = prepare ]; then
    mkdir -p "$scratch/base" "$scratch/collection"
    rsync -a "$root/models/Qwen1.5-1.8B-Chat/" "$scratch/base/"
    rsync -a "$round/" "$scratch/collection/" --exclude=checkpoints --exclude=logs
    if [ -n "$previous" ]; then
        rsync -a "$previous/" "$scratch/previous/"
    fi

    resume=$(find "$round/checkpoints" -maxdepth 2 -name complete -type f 2>/dev/null | sort | tail -1 || true)
    if [ -n "$resume" ]; then
        rsync -a "$(dirname "$resume")/" "$scratch/resume/"
    fi

    echo "Prepared node=$(hostname) scratch=$scratch"
    exit 0
fi

echo "Launcher node=$(hostname) devices=${CUDA_VISIBLE_DEVICES:-unset} workers=$workers"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

checkpoint_args=()
if [ "$stage" = collect ]; then
    if [ -n "$previous" ]; then checkpoint_args=(--checkpoint "$scratch/previous"); fi

    exec "$python" -u -m torch.distributed.run --standalone --nproc-per-node="$workers" \
        -m multi_llm.collect --base "$scratch/base" \
        --data "$scratch/collection/rows.json" --output "$scratch/collection/generated" \
        --publish "$round" --max-tokens "$max_tokens" "${checkpoint_args[@]}"
elif [ "$stage" = train ]; then
    if [ -n "$previous" ]; then checkpoint_args=(--previous "$scratch/previous"); fi
    if [ -f "$scratch/resume/complete" ]; then
        checkpoint_args=(--resume "$scratch/resume")
    fi

    exec "$python" -u -m torch.distributed.run --standalone --nproc-per-node="$workers" \
        -m multi_llm.train --base "$scratch/base" \
        --collection "$scratch/collection/training" --work "$scratch/checkpoints" \
        --checkpoints "$round/checkpoints" --config "$scratch/collection/training_config.json" \
        "${checkpoint_args[@]}"
else
    echo "Unknown stage: $stage" >&2
    exit 2
fi
