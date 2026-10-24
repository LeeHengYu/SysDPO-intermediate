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

For a pool of 512 questions, collected in four non-overlapping 128-question rounds:

```sh
~/.local/bin/uv run --no-sync python -u -m multi_llm.pipeline \
  --root /hpchome/hengyulee/sysdpo \
  --run train512 \
  --questions 512 \
  --round-size 128 \
  --max-tokens 512 \
  --workers 8
```

For the entire training split, use a new run name with `--questions all`. The final collection round can contain fewer questions than `--round-size`.

The coordinator must remain running while stages execute. To detach it from the SSH terminal:

```sh
nohup ~/.local/bin/uv run --no-sync python -u -m multi_llm.pipeline \
  --root /hpchome/hengyulee/sysdpo \
  --run full_train \
  --questions all \
  --round-size 128 \
  --max-tokens 512 \
  --workers 8 \
  > /hpchome/hengyulee/sysdpo/full_train.coordinator.log 2>&1 < /dev/null &
```

## CLI options

| Option | Meaning | Default |
|---|---|---|
| `--root PATH` | Persistent models, data and runs root | Required |
| `--run NAME` | Output directory name under `runs/` | Required |
| `--questions N` / `--questions all` | Select up to N questions, or all remaining training questions | `1024` |
| `--round-size N` | Fixed questions per independent collection round; final dataset fragment may be smaller | `128` |
| `--from-run NAME` | Restore the latest completed training round of another run | None |
| `--start-offset N` | Zero-based position in the saved training split | After source round, or `0` without a source |
| `--max-tokens N` | Maximum new tokens per generated intermediate output | `512` |
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
  --round-size 128 \
  --max-tokens 512 \
  --workers 8
```

Use a new output run name. The source must have a completed training round and a durably published checkpoint. A/B adapters, optimizer and scheduler state are restored; base/reference weights remain fixed. Training configuration is inherited, and a conflicting `--config` is rejected. The chosen checkpoint is pinned in the new run's manifest.

Automatic positioning advances by collected questions, including questions filtered out of training. For shuffled runs these are positions in the deterministic shuffled order, not the dataset's row IDs. Use `--questions all` to process the remaining questions.

## GPU stages and batching

Each collection round executes:

1. Submit a collection job. Frozen A generates three candidates per question with grouped DBS: three beams/groups, diversity penalty 10, deterministic decoding, and the configured output-token limit.
2. Finish the collection job and release its GPUs.
3. Judge the three candidate pairs through `microsoft/microsoft-decision-1` on the management node, with bounded concurrency. Responses and usage are saved in `calls/`.
4. Submit a new training job. B scores the dataset's fixed chosen/rejected final answers under each intermediate; it does not generate final answers. A and B adapters are updated.
5. Finish checkpoint synchronization, exit the training job and release its GPUs. The next round collects with updated A.

Incomplete or empty candidate triples are excluded from training. Complete duplicate or cyclic triples retain system-scoring rows but supply no intermediate preference pairs. Inspect `judging_summary.json` for actual usable counts.

`script/job.sh` defaults to one GPU worker using account `research`, partition `gpuh` and QOS `qos_research`. Each task gets one H100 GPU and four CPUs; host memory is 64 GB per node. `script/worker.sh` stages inputs on the allocated node, then `srun` starts a launcher and `torchrun` starts one Python worker per GPU. The coordinator requests 30 hours for collection and four hours for training. Increasing collection size does not increase those time limits. The allocation remains held until all workers finish and the Slurm script exits, including checkpoint-copy completion.

Set Slurm's standard `SBATCH_NTASKS` environment variable to the requested GPU count and `SBATCH_NODES=1` when starting the coordinator; for four GPUs, use `SBATCH_NTASKS=4 SBATCH_NODES=1`. The launcher requires all workers on one node. Every stage in that coordinator inherits these settings.

Worker count is fixed when the coordinator is invoked; it does not automatically downgrade from two GPUs to one. Set `SBATCH_NTASKS=1 SBATCH_NODES=1` to request one GPU on one node explicitly.

Check the partition's node limit with `scontrol show partition gpuh`. Both the current launcher and the cluster's `MaxNodes=1` limit require the requested GPUs to fit on one node.

Each fixed collection round supplies at most one optimizer update. Default training configuration:

| Setting | Value |
|---|---:|
| Cumulative optimizer update budget | 70 |
| Maximum global question rows per optimizer update | 64 |
| Checkpoints | After each collection’s single update |
| Learning rate | `6.4e-6` |
| Warmup optimizer steps | 20 |
| DPO beta | 0.5 |
| LoRA rank / alpha | 16 / 16 |
| LoRA dropout | 0.05 |
| Seed | 42 |

The selected dataset order is shuffled once using the training seed before collection; the manifest records that order and resumes reuse it. `--start-offset` addresses positions in this shuffled order. Continuing a legacy unshuffled run retains its original order.

Each collection round takes the next 128 questions in the shuffled order, regardless of which rows were trained in earlier rounds. There are no fill collections or carried rows. After judging, if more than 64 usable system rows remain, select 64 uniformly at random without replacement; otherwise use every available row in one smaller update. An empty usable set produces no update. A usable row is one saved in `scoring_requests.json`; rows without accepted intermediate pairs can still contribute system loss.

Selection uses the training seed and round name, and is saved in `training/selection.json` along with the usable count and selected row IDs. `training/scoring_requests.json` and `training/intermediate_preferences.json` contain only the selected rows and their preference pairs. Resume reuses this selection. Unselected usable rows remain available for audit but are never carried into another round. Each selected batch is split across the GPU workers, and the loss is normalized using its actual row/pair counts.

The run stops at 70 cumulative optimizer updates, including the existing 20-step warmup. At most 4,480 question rows are used, and smaller batches reduce that total. Empty rounds do not count toward the cap. RMSprop and scheduler state continue across rounds. No further collection or judging is launched after the cap. Checkpoints are synchronized at the end of every training stage, including empty stages that retain the current model weights.

Collection saves three policy candidates and two separate frozen-reference candidates per question. The latter are generated with the A adapter disabled, using the same prompt and token cap. Both sets and token counts are persisted in the question/token records. Only policy candidates are judged. `L_sys` scores the policy set with current A/B and the reference set with the frozen base models. `L_A` scores the same judge-labeled `y_chosen`/`y_rejected` under current A and the original frozen base A. Reference candidates do not enter intermediate preference pairs. Capped reference outputs are retained for scoring; empty reference outputs exclude that question from training.

Per-update audit records in `checkpoints/metrics/step_<step>.json` contain raw `L_sys`, raw pair-averaged `L_A`, `L_A_weighted = 0.5 * L_A`, `L_total`, batch rows and pair counts. These are losses computed before that update. Checkpoint `state.json` also contains `last_audit`; stdout and `--status` expose the components. Metrics newer than the latest complete checkpoint are provisional and are overwritten if those updates are replayed.

Use a fresh run name for this configuration. Historical checkpoints are not converted to the new capped, shuffled schedule, and their component losses cannot be recovered from total loss alone.
Collections without the separate reference candidates must be recollected; their scoring requests are not accepted by the new schema. Use a fresh run for this fixed-round selection policy; do not reuse a previous run’s prepared training inputs.

For a fresh run, provide a partial configuration JSON with `--config PATH`, for example:

```json
{
  "max_steps": 70,
  "optimizer_rows": 64,
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

Run artifacts live under `--root/runs/<run>/round_<index>/`. Checkpoints are stored in `checkpoints/rows_<count>/`, containing `A_adapter/`, `B_adapter/`, training state and metadata. Keep the base model and tokenizer under `--root/models/`; adapters alone cannot restore the models. Keep round data, `training/selection.json` and checkpoints to reproduce the selected training rows.

GPU jobs use temporary node-local scratch and synchronize checkpoints to persistent storage before exiting. Only checkpoint directories with a `complete` marker and no `.partial` suffix are usable.

To recover an interrupted run, rerun its original CLI command with the same settings. Completed stages and successful API responses are reused. A completed training checkpoint is reused without another update; an interrupted update without a complete checkpoint is replayed. Reusing a checkpoint within the same round requires the same GPU worker count. Use `--from-run NAME` with a new `--run` name to branch from a completed round; worker count may change between completed rounds.

If a Slurm stage failed, inspect its log and scheduler state before removing its `collect_job.json` or `train_job.json` to allow resubmission. Reconcile a null job ID before resubmitting, and never resubmit an active job. Failed paid judge calls are retained and are not automatically retried.

Keep a `code/<release>/` directory while a coordinator or submitted job references it. Updating `current` does not change an already running coordinator's code. Run only one coordinator per run; it submits the GPU jobs itself.

## Evaluate a completed run

Run evaluation on the management node from `current/`:

```sh
nohup ~/.local/bin/uv run --no-sync python -u -m multi_llm.evaluate run \
  --root /hpchome/hengyulee/sysdpo --run RUN_NAME \
  --output /hpchome/hengyulee/sysdpo/evaluations/EVALUATION_NAME \
  --gpus 2 --max-tokens 256 --workers 8 \
  > /hpchome/hengyulee/sysdpo/evaluation.log 2>&1 < /dev/null &
```

Replace `RUN_NAME` with a completed training run and `EVALUATION_NAME` with an output
directory name. `--gpus` requests 1, 2 or 4 GPUs on one node; `--max-tokens` caps new
tokens per stage; `--workers` controls concurrent judge API calls. The coordinator
submits GPU generation, waits for completion, then judges on the management node.

Outputs include `evaluation.json`, `rows.json`, per-question records, `calls/`,
`trained.json`, `prompted.json`, `chosen.json`, and `summary.json`. Completion is
marked by `evaluation.ready`. Rerunning the same command reuses saved generations
and successful judgments. Changed inputs or settings require a new output directory;
failed or interrupted API calls remain saved for inspection.

## Source layout and checks

| Directory | Responsibility |
|---|---|
| `core/` | Shared schemas, prompts, model calls, likelihoods and adapter/checkpoint logic |
| `multi-llm/` (`multi_llm` package) | Input preparation, collection, judging, coordinator, training CLI and status |
| `script/` | Deployment, Linux setup and Slurm job scripts |
| `tests/` | Shared component and production workflow tests |

From a prepared repository environment:

```sh
uv run --no-sync python -m unittest discover -s tests -t .
```

Tests use mocked API calls and tiny local models; they do not submit GPU jobs or verify real-model throughput.
