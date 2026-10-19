# SysDPO-extend remote GPU workflow

Engineering instructions for deploying and running the offline GPU pipeline on the CUHK cluster. Research plans and experiment discussion live in the parent `cuhk/` directory.

## Execution environments

| Location | Work |
|---|---|
| Local Mac | Edit code, run CPU tests, deploy over SSH |
| Management node (`ssh cuhk-compute`) | Install Linux dependencies, download models/data, run the coordinator, judge through OpenRouter, submit Slurm jobs |
| Compute node | Generate candidates and update A/B adapters on GPUs without internet access |

Use `uv` and Python 3.12 or later. Each deployed release has its own Linux `.venv`; do not upload the Mac environment. Use `uv run --no-sync` for commands so the installed GPU dependencies remain available.

## Deploy from the Mac

From the local `sysdpo-extend/` directory:

```sh
bash script/deploy.sh
```

Optional host and persistent-root arguments:

```sh
bash script/deploy.sh cuhk-compute /hpchome/hengyulee/sysdpo
```

Deployment uploads source into `code/<release>/`, runs `script/setup.sh`, and updates `current` after installation succeeds. Submitted GPU jobs receive the resolved release path, so a later deployment does not change their code or environment. Models, data and run artifacts are stored outside release directories.

## Prepare the management node

```sh
ssh cuhk-compute
cd /hpchome/hengyulee/sysdpo/current

~/.local/bin/uv run --no-sync python -m multi_llm.prepare \
  --root /hpchome/hengyulee/sysdpo
```

Preparation downloads `Qwen/Qwen1.5-1.8B-Chat` and `Intel/orca_dpo_pairs`, pins their revisions in `inputs.json`, and saves the tokenizer with the model. The existing prepared split contains 12,216 training rows and 643 test rows. Only `data/train.json` is selected by the pipeline.

Judging requires `OPENROUTER_KEY` in the management process environment, or a private `~/.config/sysdpo/openrouter.json` file containing the `OPENROUTER_KEY` field. Keep that file readable only by its owner. The coordinator removes the key from the environment passed to Slurm submissions.

## Start a run

Run commands below from `/hpchome/hengyulee/sysdpo/current` on the management node.

For 512 questions in one collection round:

```sh
~/.local/bin/uv run --no-sync python -u -m multi_llm.pipeline \
  --root /hpchome/hengyulee/sysdpo \
  --run train512 \
  --questions 512 \
  --round-size 512 \
  --max-tokens 1024 \
  --workers 8
```

For the entire training split, use a new run name with `--questions all`. The final collection round can contain fewer questions than `--round-size`.

The coordinator must remain running while stages execute. To detach it from the SSH terminal:

```sh
nohup ~/.local/bin/uv run --no-sync python -u -m multi_llm.pipeline \
  --root /hpchome/hengyulee/sysdpo \
  --run full_train \
  --questions all \
  --round-size 512 \
  --max-tokens 1024 \
  --workers 8 \
  > /hpchome/hengyulee/sysdpo/full_train.coordinator.log 2>&1 < /dev/null &
```

## CLI options

| Option | Meaning | Default |
|---|---|---|
| `--root PATH` | Persistent models, data and runs root | Required |
| `--run NAME` | Output directory name under `runs/` | Required |
| `--questions N` / `--questions all` | Select up to N questions, or all remaining training questions | `1024` |
| `--round-size N` | Questions collected from one frozen A snapshot before judging and training | `1024` |
| `--from-run NAME` | Restore the latest completed training round of another run | None |
| `--start-offset N` | Zero-based position in the saved training split | After source round, or `0` without a source |
| `--max-tokens N` | Maximum new tokens per generated intermediate output | `1024` |
| `--workers N` | Concurrent management-node judge calls | `8` |
| `--config PATH` | Training settings as a JSON object | Defaults below, or inherited from source checkpoint |
| `--status` | Read saved progress and Slurm state without launching work | Off |

A run name is a directory name such as `train512`, not a Slurm job ID. An explicit `--start-offset` overrides automatic positioning and can intentionally repeat or skip data. The experiment manifest records selected row IDs, starting checkpoint, training settings, collection-round size and output-token limit. Rerunning a run requires those settings to match.

### Continue from a completed run

```sh
~/.local/bin/uv run --no-sync python -u -m multi_llm.pipeline \
  --root /hpchome/hengyulee/sysdpo \
  --run next512 \
  --from-run train512 \
  --questions 512 \
  --round-size 512 \
  --max-tokens 1024 \
  --workers 8
```

Use a new output run name. The source must have a completed training round and a durably published checkpoint. A/B adapters, optimizer and scheduler state are restored; base/reference weights remain fixed. Training configuration is inherited, and a conflicting `--config` is rejected. The chosen checkpoint is pinned in the new run's manifest.

Automatic positioning advances by collected questions, including questions filtered out of training. These positions are array offsets in `data/train.json`, not the dataset's row IDs. Use `--questions all` to process the remaining questions.

## GPU stages and batching

Each collection round executes:

1. Submit a collection job. Frozen A generates three candidates per question with grouped DBS: three beams/groups, diversity penalty 10, deterministic decoding, and the configured output-token limit.
2. Finish the collection job and release its GPUs.
3. Judge the three candidate pairs through `microsoft/microsoft-decision-1` on the management node, with bounded concurrency. Responses and usage are saved in `calls/`.
4. Submit a new training job. B scores the dataset's fixed chosen/rejected final answers under each intermediate; it does not generate final answers. A and B adapters are updated.
5. Finish checkpoint synchronization, exit the training job and release its GPUs. The next round collects with updated A.

Incomplete or empty candidate triples are excluded from training. Complete duplicate or cyclic triples retain system-scoring rows but supply no intermediate preference pairs. Inspect `judging_summary.json` for actual usable counts.

`script/job.sh` defaults to one GPU worker using account `research`, partition `gpuh` and QOS `qos_research`. Each task gets one H100 GPU and four CPUs; host memory is 64 GB per node. `script/worker.sh` stages inputs once per node, then `srun` launches the allocated Python workers with distributed rank and rendezvous settings. The coordinator requests 30 hours for collection and four hours for training. Increasing collection size does not increase those time limits. The allocation remains held until all workers finish and the Slurm script exits, including checkpoint-copy completion.

For two GPU workers, set Slurm's standard `SBATCH_NTASKS=2` environment variable when starting the coordinator. The launcher permits one or two nodes and adapts staging and worker ranks to the actual allocation; this requires no change to the coordinator's Python code. A two-node placement uses 128 GB host memory total. Every stage in that coordinator inherits the setting.

Worker count is fixed when the coordinator is invoked; it does not automatically downgrade from two GPUs to one. Set `SBATCH_NTASKS=1 SBATCH_NODES=1` to request one GPU on one node explicitly.

Check the partition's node limit with `scontrol show partition gpuh`. If `MaxNodes=1`, both GPU workers must fit on the same node. Two-node execution requires a partition that permits it and has not been verified on this cluster.

Collection-round size and optimizer batch size are independent. Default training configuration:

| Setting | Value |
|---|---:|
| Global rows per optimizer update | 64 |
| Processed rows between checkpoints | 256 |
| Learning rate | `6.4e-6` |
| Warmup optimizer steps | 20 |
| DPO beta | 0.5 |
| LoRA rank / alpha | 16 / 16 |
| LoRA dropout | 0.05 |
| Seed | 42 |

The 64 rows are processed by one GPU or split across the allocated workers (normally 32 each with two GPUs). Filtering and round boundaries can produce a smaller final update. A full batch supplies at most 192 intermediate preference pairs. Checkpoint intervals must be divisible by optimizer batch size; the final processed row is also checkpointed. RMSprop updates the adapters, and scheduler progress continues across collection rounds.

For a fresh run, provide a partial configuration JSON with `--config PATH`, for example:

```json
{
  "optimizer_rows": 64,
  "checkpoint_rows": 256,
  "learning_rate": 0.0000064,
  "warmup_steps": 20
}
```

## Inspect progress

From another management-node terminal:

```sh
cd /hpchome/hengyulee/sysdpo/current
~/.local/bin/uv run --no-sync python -m multi_llm.pipeline \
  --root /hpchome/hengyulee/sysdpo \
  --run next512 --status
```

Status reads up to eight rounds concurrently and prints each round's report in order. It reports collection counts, filtering, judge calls and recorded API costs, scheduler states/runtimes, last logged batch loss, and durable checkpoint progress. Logged progress and saved progress can differ while synchronization runs. The command loads no model libraries or API credentials, submits no jobs and writes no experiment files. If Slurm is unavailable, saved results are still reported.

Direct scheduler commands:

```sh
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,AllocTRES
```

## Storage and recovery

Run artifacts live under `--root/runs/<run>/round_<index>/`. Checkpoints are stored in `checkpoints/rows_<count>/`, containing `A_adapter/`, `B_adapter/`, training state and metadata. Keep the base model and tokenizer under `--root/models/`; adapters alone cannot restore the models.

GPU jobs use temporary node-local scratch and synchronize checkpoints to persistent storage before exiting. Only checkpoint directories with a `complete` marker and no `.partial` suffix are usable.

To recover an interrupted run, rerun its original CLI command with the same settings. Completed stages and successful API responses are reused. Training resumes from the latest complete checkpoint and requires the same GPU worker count. Use `--from-run NAME` with a new `--run` name to branch from a completed round; worker count may change between completed rounds.

If a Slurm stage failed, inspect its log and scheduler state before removing its `collect_job.json` or `train_job.json` to allow resubmission. Reconcile a null job ID before resubmitting, and never resubmit an active job. Failed paid judge calls are retained and are not automatically retried.

Keep a `code/<release>/` directory while a coordinator or submitted job references it. Updating `current` does not change an already running coordinator's code. Run only one coordinator per run; it submits the GPU jobs itself.

## Source layout and checks

| Directory | Responsibility |
|---|---|
| `core/` | Shared schemas, prompts, model calls, likelihoods and adapter/checkpoint logic |
| `multi-llm/` (`multi_llm` package) | Input preparation, collection, judging, coordinator, training CLI and status |
| `script/` | Deployment, Linux setup and Slurm job scripts |
| `tests/` | Shared component and production workflow tests; GPU recovery verification |

From a prepared repository environment:

```sh
uv run --no-sync python -m unittest discover -s tests -t .
```

Tests use mocked API calls and tiny local models; they do not submit GPU jobs or verify real-model throughput. `tests.verify_gpu` provides a separate full-size checkpoint-recovery check requiring an existing collection and GPU allocation.
