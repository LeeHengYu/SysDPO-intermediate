# SysDPO-extend

Train a two-LLM system with system-level DPO and an additional preference loss on A's intermediate outputs. A produces guidance for B; B scores the dataset's chosen/rejected final answers during training and generates final answers during evaluation.

## Code layout

| Path | Responsibility |
|---|---|
| `core/` | Validated records, prompts, judge API, likelihoods, LoRA training and checkpoints |
| `multi-llm/` | The `multi_llm` package: preparation, collection, judging, training, orchestration, status and evaluation |
| `script/` | Deployment, dependency setup and Slurm launchers |
| `tests/` | Numerical, CPU training and mocked workflow checks |

## Install

Requires Python 3.12+ and `uv`. Run from the repository directory:

```sh
uv venv --python 3.12
uv pip install -r requirements-gpu.txt
uv pip install --no-deps --config-settings editable_mode=strict -e .
```

The GPU requirements include the base requirements. Strict editable mode exposes the `multi_llm` package mapping to static analysis tools. Select `.venv/bin/python` in your editor; retain the generated `build/` directory and repeat the editable install after adding modules.

## Deploy and prepare inputs

The management node runs the coordinator and network API calls; compute nodes run offline GPU stages. The pipeline requires Slurm and persistent storage accessible to both.

Before deployment, adapt the Slurm directives in `script/job.sh` and `script/evaluate.sh`, GPU resource flags in the launch scripts, and scratch paths in `script/worker.sh` and `script/evaluate.sh` to your cluster.

From the source checkout, provide your SSH destination and an absolute persistent project root:

```sh
REMOTE_HOST=user@management-host
PROJECT_ROOT=/shared/sysdpo
bash script/deploy.sh "$REMOTE_HOST" "$PROJECT_ROOT"
```

Deployment creates `code/<release>/`, installs its environment, then updates `current`. Existing coordinators and submitted jobs keep their resolved release paths.

On the management node:

```sh
export PROJECT_ROOT=/shared/sysdpo
cd "$PROJECT_ROOT/current"
export OPENROUTER_KEY='your-api-key'
uv run --no-sync python -m multi_llm.prepare --root "$PROJECT_ROOT"
```

Preparation downloads and pins `Qwen/Qwen1.5-1.8B-Chat` and `Intel/orca_dpo_pairs`, including the tokenizer, then saves a seeded 95%/5% train/test split. The coordinator also accepts the API key from `~/.config/sysdpo/openrouter.json` as an `OPENROUTER_KEY` field. GPU submissions exclude the key.

## Train

Run from the installed source directory on the management node:

```sh
SBATCH_NTASKS=4 SBATCH_NODES=1 \
uv run --no-sync python -u -m multi_llm.pipeline \
  --root "$PROJECT_ROOT" --run train \
  --questions all --round-size 128 --max-tokens 512 --workers 8
```

`SBATCH_NTASKS` sets the GPU worker count for collection and training. All workers must fit on one node. Keep the coordinator running, or detach the command with `nohup` and redirect its output to a log.

| Option | Meaning | Default |
|---|---|---|
| `--root PATH` | Persistent model, data and run root | Required |
| `--run NAME` | Directory name under `runs/`, not a Slurm job ID | Required |
| `--questions N` or `all` | Questions to select, subject to the update cap | `1024` |
| `--round-size N` | Questions collected per independent round | `128` |
| `--max-tokens N` | New-token cap per intermediate | `512` |
| `--workers N` | Concurrent judge API calls | `8` |
| `--config PATH` | Partial training configuration JSON | Built-in defaults or inherited configuration |
| `--from-run NAME` | Continue from another run's latest completed round | None |
| `--start-offset N` | Position in the deterministic dataset order | `0`, or inferred from source run |
| `--status` | Report saved progress and Slurm state without launching work | Off |

Each round:

1. Collect three current-policy and two fixed-reference DBS candidates per question on GPUs.
2. Release the collection allocation and judge the three policy pairs using `microsoft/microsoft-decision-1`.
3. Keep all three intermediate preferences only when their directed graph is acyclic. Duplicate/cyclic triples retain system-scoring rows; incomplete policy triples and empty reference outputs are excluded.
4. Randomly select at most 64 usable question rows and their accepted pairs. Perform one A/B update with `L_sys + 0.5 * L_A`; `L_A` updates only A. Reference models remain fixed at the initial base weights.
5. Publish the checkpoint, release the training allocation, and collect the next nonoverlapping round with updated A. Unselected rows are not carried forward.

Defaults are 70 cumulative updates, RMSprop at `6.4e-6`, 20 warmup steps, DPO beta `0.5`, LoRA rank/alpha `16/16`, dropout `0.05`, and seed `42`. Empty usable rounds perform no update. The pipeline stops collecting after the update cap, even with `--questions all`.

Override training settings with `--config config.json`, for example:

```json
{"max_steps": 70, "optimizer_rows": 64, "learning_rate": 0.0000064}
```

Prompts are in `core/prompts.json`. Use a fresh run when changing prompts or saved run settings.

## Status and recovery

```sh
uv run --no-sync python -m multi_llm.pipeline \
  --root "$PROJECT_ROOT" --run train --status
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,AllocTRES
```

Round artifacts are under `runs/<run>/round_<index>/`: generated questions, cached judge calls, selected training inputs, checkpoints, and component-loss metrics. Checkpoints contain A/B adapters, optimizer/scheduler state and RNG state; restoring a model also requires the saved base model. Only checkpoints with a `complete` marker are usable.

Rerun the original command to reuse completed stages. Only one coordinator may operate on a run. Inspect failed jobs before removing their stage job metadata for resubmission; never resubmit an active job. Saved failed API calls require inspection, while HTTP 429 responses retry after a 30-second backoff.

To branch from a completed round:

```sh
SBATCH_NTASKS=4 SBATCH_NODES=1 \
uv run --no-sync python -u -m multi_llm.pipeline \
  --root "$PROJECT_ROOT" --run continued --from-run train \
  --questions all --round-size 128 --max-tokens 512 --workers 8
```

This restores adapters and optimizer/scheduler state and advances past collected questions, including unused rows. Training configuration is inherited. `--start-offset` explicitly overrides that position. Same-round checkpoint reuse requires the original worker count; it may change between completed rounds. Keep release directories while processes or submitted jobs reference them.

## Evaluate

On the management node:

```sh
uv run --no-sync python -u -m multi_llm.evaluate run \
  --root "$PROJECT_ROOT" --run train \
  --output "$PROJECT_ROOT/evaluations/train" \
  --gpus 2 --max-tokens 256 --workers 8
```

Evaluation excludes test questions present in the training split, submits GPU generation for tuned and untuned systems, then judges tuned-versus-chosen, tuned-versus-untuned, and untuned-versus-chosen comparisons. `--gpus` accepts 1, 2 or 4 workers on one node.

Outputs include the pinned manifest, generated answers, cached calls, `summary.json`, and `evaluation.ready`. Rerunning reuses saved work; changed inputs, checkpoint or settings require a new output directory.

## Tests

```sh
uv run --no-sync python -m unittest discover -s tests -t .
```

Tests use tiny local models and mocked API/scheduler calls. They do not submit cluster jobs or measure real-model throughput.
