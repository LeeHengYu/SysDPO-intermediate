# Remote execution preparation and implementation checklist

## Agreed setup

- Management node: internet-dependent work, dependency/model/data downloads, OpenRouter judging, CPU preparation, Slurm coordination, and result inspection.
- Compute node: offline GPU generation, likelihood computation, backpropagation, and LoRA updates.
- Persistent storage: `/hpchome/<user>`; base A/B weights remain there throughout the experiment. Keep per-round adapters and training state alongside the collected data.
- GPU-job working storage: `/mnt/slurm_scratch/job_<JOB_ID>` on the compute node's flash drives. Stage frequently accessed files there; retain their durable copies under `/hpchome/<user>`. Scratch is local disk, not host RAM or GPU VRAM.
- A and B both start from `Qwen/Qwen1.5-1.8B-Chat`, matching SysDPO's multi-LLM training script. They have independent trainable LoRA adapters; identical base weights/tokenizer files can share one persistent download.
- Collection size: 1,024 dataset questions per frozen A snapshot, with three candidates per question. This is separate from inference batch size, optimizer batch size, and gradient accumulation.
- Each training job processes that 1,024-row collection. Save a checkpoint after every 256 processed question rows and copy it to `/hpchome/<user>` using a background thread while training continues. The four checkpoints do not start new collection rounds or refresh the frozen references.
- Generate candidates with SysDPO's Diverse Beam Search (DBS), adapted from two to three candidates: `num_beams=3`, `num_beam_groups=3`, `num_return_sequences=3`, `diversity_penalty=10.0`, `do_sample=False`.
- Dataset: `Intel/orca_dpo_pairs`; confirm the exact downloaded revision and row count, expected approximately 12.9k. Establish the training/evaluation split before collection.
- Judge: `microsoft/microsoft-decision-1` through OpenRouter on the management node. Only `option1` and `option2` are available; confidence does not weight the loss.
- Loss: `L_total = L_sys + 0.5 * L_A`; both DPO betas currently 0.5. A receives both terms; B receives only the system term.
- DPO references remain fixed at the initial configuration. The frozen collection snapshot is refreshed between rounds and is a separate concept.
- Initial schedule: one pass over each collection, one training seed. Treat this as a starting schedule pending throughput and preference-yield measurements.

## Details to confirm when access is ready

These items have different owners. Server facts need remote verification; deployment/setup can run through SSH from the Mac; experiment choices can be prepared locally before access is available.

### Access inputs and cluster policy

- [x] Confirm SSH access: `ssh cuhk-compute` connects to management host `head.hpc.cluster` at `137.189.75.110` as `hengyulee`, using public-key authentication without an interactive prompt. Persistent home is `/hpchome/hengyulee`; project files are deployed under `/hpchome/hengyulee/sysdpo`.
- [x] Query scheduler account limits: account `research`, partition `gpuh`, default QOS `qos_research`; maximum wall time 36 hours. QOS per-user limits are 120 CPUs, 480GB memory and eight H100 GPUs; per-job GPU limit is eight. Our initial jobs still request one GPU. `sbatch --test-only` validates a one-GPU request without allocating a job.
- [ ] Confirm management-node policy for a long-running coordinator, such as a `tmux` session, and outbound API requests. A successful connection does not establish permission for a persistent coordinator.

### Server checks executable through SSH

- [x] Check management-node tools: `uv` 0.13.0 and uv-managed Python 3.12.15 are installed under persistent user home and available in a fresh SSH command environment. System `/usr/bin/python3` remains 3.9.25. `rsync`, `curl`, `tmux` and Slurm commands are installed; persistent home is writable NFS storage.
- [x] Check management-node internet: Qwen model metadata/config, model-weight download endpoint, Orca dataset metadata, PyPI and OpenRouter's public models endpoint respond successfully. The Decisions POST endpoint returns the expected unauthenticated HTTP 401; authenticated stateless judging subsequently passed the five-question integration run.
- [x] Install `uv` and Python 3.12 remotely: `/hpchome/hengyulee/.local/bin/uv` and `/hpchome/hengyulee/.local/bin/python3.12`. Verified `uv run --no-project --offline --python 3.12 python ...`, including SSL and SQLite imports, in a fresh SSH connection.
- [x] Create the project's Linux `.venv` with `uv venv --python 3.12` during deployment, then install its dependencies. Each immutable deployed release owns its `.venv`.
- [ ] Confirm personal storage quota: `df` reports shared-filesystem space, not the user's allowance; the remote `quota -s -v` query returned no quota rows. The guide's 1000GB allowance is not yet verified through account-specific quota output.
- [x] Run a small allocated compute-node probe to record GPU model, usable VRAM, CUDA driver and runtime compatibility. Management-node inspection alone cannot establish compute-node hardware. Do not infer GPU speed from 80GB VRAM.
- [x] Use that probe to verify that a file written under `/hpchome/<user>` on management is readable from compute, and that the remotely created Python environment is usable there.
- [x] Configure `OPENROUTER_KEY` separately on management, then verify that the judge process can read it. Do not include the key in project files, deployment archives, command arguments, or logs.

### Verification record — 2026-10-13

Checks were executed over SSH on management. Slurm advertises three GPU nodes with eight H100 GPUs each; exact GPU VRAM and driver/runtime compatibility still require an allocated compute probe. `OPENROUTER_KEY` was absent from the inspected SSH environment. The temporary home-write probe was removed.

Subsequently installed `uv` 0.13.0 using the [official installer](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.12.15 using `uv python install 3.12`. Both work in a fresh SSH session, including an offline interpreter check. System Python and shell profiles were not changed. No project deployment/dependency installation, model-weight download, paid judge call or actual GPU job was performed; compute-node access to these installations remains to be tested.

### Experiment configuration prepared locally

- [x] Pin the exact `Qwen/Qwen1.5-1.8B-Chat` checkpoint/tokenizer revision for both A and B.
- [x] Select LoRA rank, alpha, dropout, and target modules by referring to the local SysDPO configuration. Record any changes necessary for our chosen models.
- [x] Prepare a pinned train/test split, prompt grouping/order, learning rate and initial sequence limits in configuration files; no separate validation set is used. Choose effective optimizer and inference batch sizes after the GPU pilot.
- [x] Prepare dependency versions and the agreed DBS configuration locally; verify grouped DBS support in the remote environment and pin the version that passes. Keep decoding settings constant during the experiment.

## 1. Deploy from the Mac

- [x] Prepare a small `deploy.sh` using `rsync` over SSH, with remote host and directory as arguments.
- [x] Have the local script execute remote setup commands through `ssh`: create the persistent directories, run management-node checks, and invoke an uploaded setup script. It must be runnable from this Mac without manually copying files or entering each setup command remotely.
- [x] Upload code, prompt/configuration files, dependency declarations, and submission scripts. Exclude the Mac `.venv`, caches, credentials, model weights, and run artifacts by default.
- [x] Preserve remote model/data/run directories on subsequent deployments; do not use deletion against the whole remote project tree.
- [x] Record the deployed source version or file hashes in each experiment. Keep running jobs on a stable deployed copy when newer code is uploaded.
- [x] Create a Linux `.venv` on the management node with `uv`; install packages there. A macOS virtual environment must not be copied for execution on Linux.
- [x] Add tested GPU dependencies to the remote requirements: Transformers, PEFT, Accelerate and any selected quantization library. Resolve PyTorch/CUDA compatibility against the actual node.
- [x] Prepare a small remote `setup.sh` for `uv` environment creation, dependency installation, and the model/dataset downloads in Section 2. Keep credentials separate and reuse existing persistent downloads on subsequent deployments.
- [x] Save a setup report with the discovered server settings and installed versions. Submit the small compute probe once partition/account settings are confirmed; deployment itself must not start the full experiment.

The deploy/setup scripts and experiment configuration can be written before remote access is ready. Once access is available, supply the SSH/account values, run management checks/setup, run the compute probe, and then validate the five-question workflow in Section 7. Model downloads and Linux installation execute remotely through SSH; the Mac uploads their instructions and project files.

## 2. Prepare persistent inputs

- [x] Download exact base A/B weights, tokenizers and configuration files to local directories under persistent storage.
- [x] Use one shared Qwen base directory for both roles, with separate A/B adapter directories. Sharing files does not tie their adapter parameters or updates.
- [x] Download the full dataset and save it locally, with revision, row IDs and split assignments. Compute jobs must not invoke a Hub download.
- [x] Preserve initial reference adapter state if references include adapters. If references are the plain base models, record that explicitly.
- [x] Confirm model loading with local paths and `local_files_only=True`; use offline Hub/dataset settings in compute jobs.
- [x] Prepare Linux dependencies on the management node before starting compute jobs, including any optional kernels required by the selected runtime.

Suggested persistent layout:

```text
/hpchome/<user>/sysdpo/
    code/<deployment_id>/
        .venv/
    models/Qwen1.5-1.8B-Chat/        # shared immutable base files for A and B
    inputs.json                    # pinned base-only reference identity
    data/train.json
    data/test.json
    runs/<experiment_id>/
        experiment.json
        round_000/
            collection_config.json
            question_<row_id>.json
            candidates.ready
            calls/<row_id>_judge_<pair>.json
            intermediate_preferences.json
            scoring_requests.json
            preferences.ready
            logs/
            checkpoints/rows_000256/
                A_adapter/
                B_adapter/
                training_state.pt
                state.json
                complete
            checkpoints/rows_000512/
            checkpoints/rows_000768/
            checkpoints/rows_001024/
```

### GPU-job scratch and checkpoint persistence

The [cluster scratch-space guide](https://github.com/CUHK-Business-School-AI-Hub/ai_hub_HPC_user_guide#local-scratch-space-on-gpu-nodes) documents a 1000GB job-local flash directory, faster I/O than `/hpchome`, immediate deletion at job exit, and regular incremental synchronization of important data back to `/hpchome`.

- [x] At each collection/training job start, stage the shared base model, tokenizers, required adapters/training state, and that round's inputs from persistent storage into `/mnt/slurm_scratch/job_${SLURM_JOB_ID}`. Each new job has its own scratch directory; never depend on the previous job's scratch.
- [x] Point GPU processes at the staged files. Write temporary/tokenized data, generated candidates and checkpoint files to scratch first. Keep scratch usage within its quota.
- [x] Incrementally copy completed candidate records and important outputs to the persistent round directory. Disposable caches need not be copied back.
- [x] At 256, 512, 768 and 1,024 processed question rows, finish writing a consistent A/B adapter and training-state snapshot in a separate scratch checkpoint directory. Save at a completed optimizer-update boundary; choose optimizer batches/accumulation that align with these row boundaries. Count question rows, not the three generated candidates or accepted preference pairs.
- [x] Submit each completed snapshot to one background copy thread. Training continues during transfer to the corresponding checkpoint directory under `/hpchome/<user>`. The thread copies saved files, not live model/optimizer objects; the local snapshot must remain unchanged until copying finishes.
- [x] Keep transfers sequential through that worker and check their results. Publish each persistent checkpoint's `complete` marker only after its copy succeeds. Failed transfers must not be marked complete or silently ignored.
- [x] For a smaller final collection or pilot, also save and synchronize its final checkpoint even when its row count is not a multiple of 256. Before normal job exit, wait for all background transfers to finish successfully, then publish the training-stage completion marker.
- [x] Copy directly from the compute job to its mounted `/hpchome/<user>` path. This is persistent storage accessible to management; it does not require SSH into management or compute-node internet access.
- [x] Request 30 hours for collection and four hours for training, including staging/checkpoint synchronization; the 1,024-row pilot must verify these budgets. Regular saves are the recovery mechanism; a final copy on exit alone cannot protect against abrupt termination.
- [x] Write stage-ready markers under persistent storage only after the corresponding outputs are copied successfully. Management judging and the next GPU job consume these durable files.

## 3. Implement offline GPU collection

- [x] Add `collect.py` to load base A plus the previous round's A adapter; round zero uses the agreed initial A configuration.
- [x] Freeze that snapshot for the entire collection and switch to inference mode. Never update A while generating this round's candidates.
- [x] Generate `y1`, `y2`, `y3` for each assigned question, one question per generation call. The last collection may be smaller.
- [x] Reuse SysDPO's grouped DBS generation pattern with three beams/groups/returned sequences and diversity penalty 10.0. Save the decoding configuration in the manifest; do not inherit the upstream hard-coded two-candidate reference generation.
- [x] Check EOS/completion status and retained candidate IDs. DBS can still produce duplicate or near-equivalent outputs, so retain the existing duplicate and preference-cycle checks. Select a sufficient generation limit from the pilot rather than blindly adopting SysDPO's 200-token default.
- [x] Save exact texts, dataset system instruction/question, `z_chosen`, `z_rejected`, completion status, token counts, prompts, row/candidate IDs, and collection checkpoint identity.
- [x] Preserve partial outputs for inspection; exclude incomplete triples from normal scoring/label collection. Do not silently truncate the task or candidate.
- [x] Resume already saved candidates rather than regenerate a completed collection on restart.
- [x] Write `candidates.ready` only after validating the completed collection manifest and outputs and copying them from scratch to persistent storage.
- [x] Add `job.sh` collection stage with the cluster's resource settings, repository environment activation, local model paths, and log locations.

## 4. Implement management-node judging

- [x] Add `judge.py` to consume a completed collection and call the existing `call_judge(state, rubric)` independently of A generation.
- [x] Parallelize management-node API judging with a bounded, configurable number of concurrent requests. Questions and their three independent comparisons can run concurrently; choose the limit from provider rate limits and the pilot. Do not judge the 1,024 questions serially.
- [x] Keep one stateless request per pair with randomized recorded presentation order. Save raw API responses and any returned probabilities, usage or cost, plus errors.
- [x] Check each triple for duplicate texts before scheduling its comparisons; duplicates discard its intermediate preferences without paid judge calls. Record unexpected labels or API failures without automatic retry.
- [x] Associate each result with its question/pair ID and recorded presentation order, regardless of completion order. Gather all three judgments for a nonduplicate triple before checking its graph; concurrency does not change preference acceptance.
- [x] For three strict judgments, use `topological_sort` to accept all three preference edges only when they are acyclic. Discard cyclic triples.
- [x] Preserve complete system-scoring rows even when intermediate comparisons are discarded. Mask repeated candidate texts for system marginalization.
- [x] Reuse completed API records on restart and stop for unresolved request failures, rather than silently submit an incomplete collection for training. Any rerun of a failed paid request must be explicit.
- [x] Write `preferences.ready` after validating the complete judging/export stage. Zero accepted pairs is a recorded collection outcome, not an API failure; report it before advancing.

## 5. Implement offline LoRA training

Loss routing:

| Loss | Supervision | A adapter | B adapter |
| --- | --- | --- | --- |
| `L_sys` | Dataset final-answer pair `z_chosen`, `z_rejected`, scored through the saved intermediate candidates | Updated | Updated |
| `0.5 * L_A` | Judge-labeled intermediate pairs `y_chosen`, `y_rejected` | Updated | No gradient |

The system score combines differentiable current A scores of each `y_i` with current B scores of each dataset answer conditioned on that `y_i`, then marginalizes over candidates. Thus final-answer preferences can adjust both adapters. Intermediate DPO uses only A's current/reference scores; B is absent from this term. Base weights and initial references stay frozen. Judge calls and saved candidate generation are outside autograd; teacher-forced likelihoods are recomputed during training.

- [x] Add actual model loading and optimizer updates in `core/training.py`.
- [x] Restore base A/B plus the previous round's adapters and optimizer/scheduler state. Keep the optimizer step count continuous across rounds.
- [x] Load fixed references with their initial adapter state or the base-only configuration. Never overwrite reference adapters with current adapters.
- [x] Compute differentiable current A likelihoods of saved candidates and current B likelihoods of dataset chosen/rejected answers conditioned on each candidate. No generated final `z` is needed for this loss.
- [x] Compute reference likelihoods under `no_grad`. Cache these only when reference weights, texts, tokenization and scoring convention match exactly.
- [x] Preserve completion/EOS masking, B synthesis instructions, and the chosen mean-versus-sum scoring convention from `likelihood.py`.
- [x] Implement minibatching with independent normalization: system loss over questions, auxiliary loss over accepted pairs. Do not silently change the relative weighting when preference counts vary or microbatches are short.
- [x] When a collection has zero accepted pairs, run and label its system-only updates explicitly; track auxiliary supervision across the experiment.
- [x] Save A/B LoRA adapters, their configurations, base revisions, tokenizer identities, optimizer/scheduler state, random states, step and consumed-row position.
- [x] Implement the 256-question-row checkpoint cadence and background copy thread described in Section 2 within each 1,024-row training job. Include the exact consumed-row position in each snapshot for resumption.
- [x] Add `job.sh` training stage; create a completion marker only after a complete, loadable checkpoint has reached persistent storage.

## 6. Implement the management-node coordinator

- [x] Add `pipeline.py` to partition training rows into 1,024-row collections and coordinate one round at a time.
- [x] Submit generation through Slurm, wait for success and `candidates.ready`, then run parallel API judging on the management node.
- [x] After `preferences.ready`, submit the GPU training job automatically. Wait for a successful saved checkpoint before advancing to the next round.
- [x] Use `sbatch --parsable` plus `sacct` polling; record job IDs, exit status and log paths. A submission or stage failure stops advancement without deleting prior artifacts.
- [x] Resume the last completed stage and training position after a coordinator or job interruption. Do not submit a second active job for an already submitted stage on restart.
- [x] Keep API calls outside GPU allocations. Release the collection GPU job before judging starts.
- [x] Use job-local scratch for GPU-job file I/O, following the staging and synchronization procedure in Section 2. Advance only from durable outputs/checkpoints in `/hpchome`.

```text
for each collection round:
    collect(A_current, assigned_rows)          # offline GPU job
    judge(saved_candidates)                   # management node, OpenRouter
    train(A_current, B_current, fixed_refs)    # offline GPU job
    save adapters + training state
    use the saved A/B checkpoint for the next round
```

## 7. Validate before the full experiment

- [x] Run existing offline tests in the remote Linux `.venv`.
- [x] Run five questions through real local A generation, management-node judging and actual GPU weight updates. Verify expected adapter parameters changed and references did not.
- [ ] Reload the saved A/B adapters and reproduce evaluation-mode logits on a fixed input within numerical tolerance; then generate from the restored A for the next round.
- [ ] Exercise resume once: no repeated successful judge calls, no repeated optimizer updates, correct next dataset rows and checkpoint.
- [ ] Verify scratch staging and checkpoint copy-back with the pilot: after the job exits and scratch is deleted, management can read candidates and a fresh job can restore the durable checkpoint.
- [ ] Verify all four checkpoints in the 1,024-row pilot, that training continues during copying, and that job completion waits for the last transfer. Resume from an intermediate checkpoint and check the next row and optimizer state.
- [ ] Run a 1,024-question pilot and measure generation time, judge time/cost, accepted-pair yield, queue time, peak GPU memory, training throughput and checkpoint size.
- [ ] Choose optimizer/inference minibatches from measured GPU memory; update time estimates using the actual GPU rather than an assumed H100.
- [ ] Implement held-out evaluation: intermediate quality and robustness/contribution tests, plus final system generation and preference evaluation. Keep evaluation prompts out of training and collection.

## Current implementation and verification status

Checked implementation items have been implemented; the validation section distinguishes tests already run from the remaining full-size pilot and research evaluation.

- [x] Implemented Linux deployment/setup, pinned Hugging Face downloads, offline DBS collection, concurrent management-node API judging and Slurm coordination.
- [x] Implemented real Qwen A/B LoRA weight updates in `core/training.py`, fixed base references, independently normalized system/intermediate DPO losses and completion/EOS scoring.
- [x] Implemented immutable scratch snapshots, background persistent copies, optimizer/scheduler/RNG restoration and continuation from the previous round.
- [x] Renamed `hpc/` to `multi-llm/`, extracted reusable code into the `core` package and removed process-path modifications. Each deployment now owns its environment and package installation.
- [x] Local suite: 31 tests discovered (30 pass; CUDA-only case skipped on Mac), including exact resumed tiny-Qwen adapter weights, gradient replay with dropout and concurrent pair attribution/cycle/duplicate handling.
- [x] Remote management dependencies, exact model/dataset downloads and GPU allocation tested. Probe job 8574 used NVIDIA H100 NVL with 95,830 MiB reported memory, Torch 2.8.0+cu128 and BF16 support.
- [x] First five-question integration passed: collection 8575, training 8576. Three complete question rows produced nine accepted pairs and three real optimizer steps. Both adapters changed. Two rows reached the old 2,048-token cap and were excluded; the default cap is now 8,192. Finite generation limits remain necessary; incomplete outputs are never trained.
- [x] Verified the refactored package deployment on Linux: collection 8577 and training 8579 completed; management-node workflow tests pass.
- [x] Verified full-size single-GPU checkpoint resume and restored A generation (job 8581). With deterministic SDPA math, resumed A/B adapters match uninterrupted training exactly; default fused CUDA kernels can differ numerically. CUDA BF16 replay-gradient checks pass (job 8582).
- [ ] Run a complete 1,024-question pilot to measure throughput, completion/acceptance yield, memory and all four checkpoint boundaries. The five-question pilot uses optimizer rows 1 / checkpoint rows 2 / no warmup; main defaults remain 64 / 256 / 20.
- [ ] Implement the planned held-out quality, robustness/contribution and final generation evaluation; the saved 643-row test split is not yet evaluated.

Downloaded inputs: Qwen revision `e482ee3f73c375a627a16fdf66fd0c8279743ca6`; Orca revision `624952e3f420ae18d88b31977ee2ea436c833abb`. There are 12,859 rows, split with seed 42 into 12,216 training and 643 test rows.

Entry points and operational instructions: [README.md](README.md). Completed deployments and test artifacts stay under `/hpchome/hengyulee/sysdpo/`.

Cluster instructions: [AI Hub HPC user guide](https://github.com/CUHK-Business-School-AI-Hub/ai_hub_HPC_user_guide). Confirm account-specific limits and filesystem behavior at initial setup.

Locally checked SysDPO sources: [model selection](/Users/hylee-mba/Documents/projects/SysDPO/scripts/train/train_multi_llm_sysdpo.sh:7), [DBS defaults](/Users/hylee-mba/Documents/projects/SysDPO/sysdpo/multi_llm/train.py:87), and [DBS generation](/Users/hylee-mba/Documents/projects/SysDPO/sysdpo/multi_llm/utils.py:413). Three candidates are our adaptation; the released default is two.

## Two-GPU follow-up and readability audit

- [x] Requests now use two GPUs per job. Questions are split between A/B/reference replicas; global optimizer batch stays 64. Gradients are summed before updates, and checkpoints store per-worker RNG states.
- [x] Added blank lines between logical blocks throughout the `multi-llm/` Python and shell entry points. Spacing-only edits preserve Python ASTs and shell tokens.
- [x] Fixed the process-group timeout for uneven DBS shards; local and remote management-node tests pass.
- [x] Completed bounded two-GPU training/resume/continuation verification (job 8583, two minutes, exit 0). Both adapters resume exactly under deterministic SDPA math; reload scores match, restored generation completes and optimizer steps continue from 2 to 4. Allocation released.

Two-GPU wall-time figures in `bottleneck_investigation.md` are conditional on scaling; total GPU-hours do not halve when using two GPUs. A full 1,024-row throughput pilot remains separate work.
