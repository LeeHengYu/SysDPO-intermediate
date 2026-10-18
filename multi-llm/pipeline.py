"""Management-node orchestration; GPU jobs never invoke network APIs."""

import argparse
import fcntl
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import time

from multi_llm.judge import judge_collection
from core.training import TrainingConfig
from core.workflow import save_json


def run_job(stage, root, directory, previous=None):
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
        result = subprocess.run(["sbatch", "--parsable", "--time",
            "30:00:00" if stage == "collect" else "04:00:00", "--output", str(directory / f"{stage}_%j.log"),
            str(script), stage, str(root), str(directory), str(previous) if previous else "", str(script.parent.parent)],
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


def pipeline(root, run, questions, workers, config, round_size=1024):
    root = root.resolve()
    if round_size < 1:
        raise ValueError("Round size must be positive")

    if not os.environ.get("OPENROUTER_KEY"):
        key_file = Path.home() / ".config" / "sysdpo" / "openrouter.json"
        os.environ["OPENROUTER_KEY"] = json.loads(key_file.read_text())["OPENROUTER_KEY"]

    if not run or Path(run).name != run or run in (".", ".."):
        raise ValueError("Run must be a single directory name")

    directory = root / "runs" / run
    directory.mkdir(parents=True, exist_ok=True)

    lock = (directory / "coordinator.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    rows = json.loads((root / "data" / "train.json").read_text())[:questions]
    if not rows or questions < 1:
        raise ValueError("Need positive question count")

    manifest = {"row_ids": [row["id"] for row in rows], "training": asdict(config), "round_size": round_size}
    path = directory / "experiment.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Existing experiment uses a different configuration")

    save_json(path, manifest)

    previous = None
    for offset in range(0, len(rows), round_size):
        current = directory / f"round_{offset // round_size:03d}"
        current.mkdir(exist_ok=True)
        save_json(current / "rows.json", rows[offset:offset + round_size])
        save_json(current / "training_config.json", asdict(config))

        run_job("collect", root, current, previous)
        if not (current / "preferences.ready").exists():
            judge_collection(current, workers)

        run_job("train", root, current, previous)

        final = sorted((current / "checkpoints").glob("rows_*/complete"))
        if not final:
            raise RuntimeError("Completed training has no durable checkpoint")
        previous = final[-1].parent

    print(f"Experiment completed: {previous}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--questions", type=int, default=1024)
    parser.add_argument("--round-size", type=int, default=1024,
                        help="Question rows collected before each judging/training round")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--config", type=Path)

    args = parser.parse_args()
    config = TrainingConfig(**json.loads(args.config.read_text())) if args.config else TrainingConfig()

    pipeline(args.root, args.run, args.questions, args.workers, config, args.round_size)
