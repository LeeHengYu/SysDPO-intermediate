"""Management-node orchestration; GPU jobs never invoke network APIs."""

import argparse
import fcntl
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import subprocess
import time

from multi_llm.runs import continuation, latest_checkpoint, prepare_training, run_directory


def run_job(stage, root, directory, previous=None, max_tokens=512):
    from core.workflow import save_json

    marker = directory / ("candidates.ready" if stage == "collect" else "training.ready")
    if marker.exists():
        return

    job_file = directory / f"{stage}_job.json"
    if job_file.exists():
        job_id = json.loads(job_file.read_text())["job_id"]
        if job_id is None:
            raise RuntimeError("Submission was interrupted; reconcile Slurm jobs before resubmitting")
    else:
        script = Path(__file__).resolve().parent.parent / "script" / "job.sh"
        save_json(job_file, {"job_id": None, "stage": stage, "code": str(script.parent.parent)})
        result = subprocess.run(["sbatch", "--parsable",
            "--ntasks", os.environ.get("SBATCH_NTASKS", "1"),
            "--nodes", os.environ.get("SBATCH_NODES", "1"), "--time",
            "30:00:00" if stage == "collect" else "04:00:00", "--output", str(directory / f"{stage}_%j.log"),
            str(script), stage, str(root), str(directory), str(previous) if previous else "",
            str(script.parent.parent), str(max_tokens)],
            check=True, text=True, capture_output=True,
            env={key: value for key, value in os.environ.items() if key != "OPENROUTER_KEY"})
        job_id = result.stdout.strip().split(";")[0]
        if not job_id.isdigit():
            raise RuntimeError(f"Invalid Slurm submission response: {result.stdout}")
        save_json(job_file, {"job_id": job_id, "stage": stage, "code": str(script.parent.parent)})

    print(f"{stage}: Slurm job {job_id}", flush=True)

    while True:
        result = subprocess.run(["sacct", "-nX", "-P", "-j", job_id, "--format=State,ExitCode"],
                                check=True, text=True, capture_output=True)
        states = [line.split("|") for line in result.stdout.splitlines() if line.strip()]
        if states:
            state, exit_code = states[0][:2]
            if state == "COMPLETED":
                if exit_code != "0:0" or not marker.exists():
                    raise RuntimeError(f"Job {job_id} completed without durable output")
                return
            if state.split()[0].rstrip("+") in {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
                                                "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}:
                raise RuntimeError(f"Job {job_id} stopped: {state} {exit_code}; inspect its log")

        time.sleep(15)


def pipeline(root, run, questions, workers, config=None, round_size=128,
             from_run=None, start_offset=None, max_tokens=512):
    from multi_llm.judge import judge_collection
    from core.training import TrainingConfig
    from core.workflow import save_json

    root = root.resolve()
    if round_size < 1 or max_tokens < 1:
        raise ValueError("Round size must be positive; max tokens must be positive")
    if from_run == run:
        raise ValueError("Use a new run name when continuing from another run")

    directory = run_directory(root, run)
    directory.mkdir(parents=True, exist_ok=True)

    with (directory / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        dataset = json.loads((root / "data" / "train.json").read_text())
        path = directory / "experiment.json"
        saved = json.loads(path.read_text()) if path.exists() else None
        previous, inferred_offset, inherited = None, 0, None
        if from_run:
            pinned = saved.get("previous") if saved and saved.get("from_run") == from_run else None
            previous, inferred_offset, inherited = continuation(root, from_run, dataset, pinned)

        config = config if config is not None else TrainingConfig(**(inherited or {}))
        if inherited is not None and asdict(config) != inherited:
            raise ValueError("Training configuration differs from source checkpoint")
        ordering = saved
        if ordering is None and from_run:
            ordering = json.loads((run_directory(root, from_run) / "experiment.json").read_text())
        shuffle = ordering.get("shuffle", False) if ordering is not None else True
        if shuffle:
            random.Random(config.seed).shuffle(dataset)
        start_offset = inferred_offset if start_offset is None else start_offset
        if start_offset < 0 or start_offset >= len(dataset):
            raise ValueError("Start offset must select a row in the training split")
        count = len(dataset) - start_offset if questions == "all" else int(questions)
        rows = dataset[start_offset:start_offset + count]
        if count < 1 or not rows:
            raise ValueError("Need positive question count")

        manifest = {"row_ids": [row["id"] for row in rows], "training": asdict(config),
                    "round_size": round_size, "start_offset": start_offset, "max_tokens": max_tokens,
                    "from_run": from_run, "previous": str(previous) if previous else None, "shuffle": shuffle}
        if saved is not None:
            saved = {"start_offset": 0, "max_tokens": 1024, "from_run": None, "previous": None,
                     "shuffle": False, **saved}
            if saved != manifest:
                raise ValueError("Existing experiment uses a different configuration")

        if not os.environ.get("OPENROUTER_KEY"):
            key_file = Path.home() / ".config" / "sysdpo" / "openrouter.json"
            os.environ["OPENROUTER_KEY"] = json.loads(key_file.read_text())["OPENROUTER_KEY"]

        save_json(path, manifest)
        for offset in range(0, len(rows), round_size):
            step = json.loads((previous / "state.json").read_text())["optimizer_steps"] if previous else 0
            if step >= config.max_steps:
                break
            current = directory / f"round_{offset // round_size:03d}"
            current.mkdir(exist_ok=True)
            save_json(current / "rows.json", rows[offset:offset + round_size])
            save_json(current / "training_config.json", asdict(config))

            run_job("collect", root, current, previous, max_tokens)
            if not (current / "preferences.ready").exists():
                judge_collection(current, workers)

            prepare_training(current, config)
            run_job("train", root, current, previous, max_tokens)
            previous = latest_checkpoint(current)
            if previous is None:
                raise RuntimeError("Completed training has no durable checkpoint")

        print(f"Experiment completed: {previous}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--questions", default="1024", help="Additional question count, or all remaining rows")
    parser.add_argument("--round-size", type=int, default=128,
                        help="Fixed questions per collection round; no refill or carryover")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--from-run", help="Source run whose latest completed round supplies the checkpoint")
    parser.add_argument("--start-offset", type=int, help="Training-split position; inferred from --from-run otherwise 0")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum new tokens per intermediate output")
    parser.add_argument("--status", action="store_true", help="Read saved progress and Slurm state without launching work")

    args = parser.parse_args(argv)
    if args.status:
        from multi_llm.status import show_status
        show_status(args.root.resolve(), args.run)
        return

    from core.training import TrainingConfig
    config = TrainingConfig(**json.loads(args.config.read_text())) if args.config else None
    pipeline(args.root, args.run, args.questions, args.workers, config, args.round_size,
             args.from_run, args.start_offset, args.max_tokens)


if __name__ == "__main__":
    main()
