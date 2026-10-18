# Offline GPU workflow

The Mac uploads source and dependency declarations. The management node creates its own Linux environment, downloads pinned inputs, runs parallel API judging, and submits offline GPU jobs. Do not copy the Mac `.venv`.

```sh
bash script/deploy.sh
ssh cuhk-compute '/hpchome/hengyulee/sysdpo/current/.venv/bin/python -m multi_llm.prepare --root /hpchome/hengyulee/sysdpo'
ssh cuhk-compute '/hpchome/hengyulee/sysdpo/current/.venv/bin/python -u -m multi_llm.pipeline --root /hpchome/hengyulee/sysdpo --run pilot1024 --questions 1024 --workers 8'
```

`script/` contains `deploy.sh`, `setup.sh` and the Slurm entry point `job.sh`. `deploy.sh` creates an immutable source release and updates `current` after dependencies install. GPU jobs receive the resolved release path, so a later upload cannot change their source. Each release has its own Linux `.venv`, so installing a new release does not change running jobs.

The judge reads `OPENROUTER_KEY`, or the private user file `~/.config/sysdpo/openrouter.json` containing that key. Credentials are excluded from deployments. Successful calls are cached; failed or interrupted paid calls are not automatically retried.

`prepare.py` downloads Qwen1.5-1.8B-Chat and Orca with recorded commit revisions. The seed-42 split reserves 5% for testing. Current downloaded data has 12,859 rows: 12,216 training and 643 test. No separate validation split is currently used.

Each round defaults to up to 1,024 questions; `--round-size` changes this independently of the optimizer batch size. For the full training split in 512-row collection rounds, use `--questions all --round-size 512` with a new run name. The final collection round can be smaller. The round size is recorded in the experiment manifest and cannot change within an existing run. The collector processes every assigned row, including rounds larger than 1,024. The coordinator requests 30 hours for collection and four hours for training; each stage exits and releases its GPUs before the next stage gets a fresh allocation. The collection request reflects the observed long-output cost and stays within the cluster's 36-hour maximum. `collect.py` uses frozen current A plus grouped DBS (three beams/groups, diversity penalty 10); `judge.py` runs independent comparisons concurrently. Duplicate/cyclic preferences do not remove complete system scoring rows. Incomplete generation is recorded and excluded; inspect `judging_summary.json` for the actual usable row count.

Continue from a named run with a new output run name:

```sh
uv run --no-sync python -m multi_llm.pipeline \
  --root /hpchome/hengyulee/sysdpo --run next512 \
  --from-run pilot512_continue --questions 512 --round-size 512 --max-tokens 1024
```

`--from-run` selects the latest completed training round and restores its A/B adapters, optimizer and scheduler using the original fixed reference. It inherits that checkpoint's training configuration; a conflicting `--config` is rejected. Dataset selection starts after the source round's collected questions, including filtered-out questions. Thus continuing from `pilot512_continue` starts at training position 1,536, rather than the 1,118 usable rows. The chosen checkpoint and starting position are saved in the new run's manifest; repeating the same command keeps that checkpoint even if the parent run advances.

`--start-offset N` explicitly overrides the training-split starting position (zero-based); without a source run it defaults to zero. `--questions N` selects up to N additional questions from that position, while `--questions all` selects all remaining training rows. Held-out test data is never selected. `--max-tokens N` controls maximum new tokens per intermediate output (default 1,024), and is forwarded to collection jobs. Rerunning an existing experiment requires matching row selection, training configuration, round size and token limit.

Inspect a run from another management-node terminal:

```sh
uv run --no-sync python -m multi_llm.pipeline \
  --root /hpchome/hengyulee/sysdpo --run next512 --status
```

`--status` reads saved collection records, judge outcomes, recorded API costs, training logs and durable checkpoint metadata, and queries Slurm for the saved job IDs. It reports logged progress separately from checkpoint progress. It loads no model libraries or API credentials, writes no run files, and submits no jobs. If Slurm is unavailable, it reports saved progress with scheduler state unknown. Recorded costs can be incomplete while calls are in flight. `runs.py` handles saved run/checkpoint selection; `status.py` owns this read-only report.

`python -m multi_llm.train` loads independent A/B LoRA adapters and a fixed base reference. RMSprop defaults to an effective learning rate of 6.4e-6 (the upstream script's 1e-7 scaled by its 64-row accumulation), with 20 warmup steps. LoRA rank/alpha are 16, dropout is 0.05, and all seven attention/MLP projection modules are targeted. Each job requests two GPUs and starts two workers. Each GPU holds A, B and the fixed reference; question rows are split between workers. The global optimizer batch stays at 64 question rows (normally 32 per worker), and gradients are summed before each update. Completion graphs are processed one at a time on each worker. System and intermediate losses are normalized separately within each optimizer batch.

The score pass computes derivatives of `L_sys + 0.5 * L_A` with respect to A/B log scores. A second pass recomputes each completion and applies those derivatives to adapter parameters. It replays dropout RNG states, preserving the gradient of the same forward computation while avoiding nine simultaneously retained completion graphs. References receive no gradients. Token-mean scoring and explicit EOS follow the existing local likelihood implementation.

Snapshots at 256/512/768/1024 rows include both adapters, optimizer/scheduler state, RNG states, row position, data digest and base identity. A background worker copies completed snapshots from scratch into a persistent `.partial` directory and publishes `complete` only when copying succeeds. Job exit waits for pending transfers. Smaller test collections also save their final state.

Resume the same collection with `--resume` and the same worker count; continue a new collection with `--previous`. `job.sh` finds the latest durable checkpoint for a resumed training stage. A failed Slurm stage stops the coordinator. After inspecting the failure and confirming the old job is terminal, remove that stage's `*_job.json` to explicitly resubmit; successful API records and checkpoints remain. If submission was interrupted before its ID was saved, reconcile the scheduler before clearing the submission record. Rerunning a completed experiment skips its ready stages.

## Packages and local setup

`multi-llm/` is installed as the `multi_llm` Python package. It owns downloads, collection, judging, Slurm orchestration and the training entry point. `core/` contains shared schemas, prompts, model calls, likelihoods and optimizer/checkpoint logic. `smoke/` contains only the API smoke entry point and tests; production never imports it.

```sh
uv pip install --python .venv/bin/python -r requirements-gpu.txt
uv pip install --python .venv/bin/python --no-deps -e .
uv run --active --no-sync python -m multi_llm.train --help
uv run --active --no-sync python -m smoke.train --help
```

Run modules with `python -m`; neither entry points nor tests modify `sys.path`. Editable installation also makes the packages importable outside the checkout directory. Prompts are in `core/prompts.json` and are included as package data.

## Local and remote checks

```sh
uv run --active --no-sync python -m unittest smoke.test.test_offline
uv run --active --no-sync python -m unittest smoke.test.test_workflow
uv run --active --no-sync python -m unittest smoke.test.test_pipeline_cli
```

The first test uses tiny randomly initialized Qwen models with real LoRA, optimizer updates and checkpoint reloads; it checks gradient equivalence and exact resumed weights. It does not replace full-size GPU validation. The second checks concurrent judging, cycle rejection and duplicate prefiltering without paid calls.

The five-question remote test uses optimizer rows 1, checkpoint rows 2 and no warmup to exercise multiple updates and checkpoints quickly. These pilot overrides are not the main 1,024-row configuration. Remote full-size test results and remaining work are recorded in `remote_execution_todo.md` and `audit/implementation.md`.
