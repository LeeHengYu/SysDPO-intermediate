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

from multi_llm.runs import latest_checkpoint, prepare_training, run_directory

GPU_CHOICES = (1, 2, 4)
JUDGE_WORKERS = 8
MAX_TOKENS = 512


def run_job(stage, root, directory, previous=None, max_tokens=MAX_TOKENS, gpus=1):
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
        result = subprocess.run(["sbatch", "--parsable", "--nodes=1", f"--gres=gpu:{gpus}",
            f"--cpus-per-task={4 * gpus}", "--time",
            "30:00:00" if stage == "collect" else "04:00:00", "--output", str(directory / f"{stage}_%j.log"),
            str(script), stage, str(root), str(directory), str(previous) if previous else "",
            str(script.parent.parent), str(max_tokens), str(gpus)],
            check=True, text=True, capture_output=True,
            env={key: value for key, value in os.environ.items() if key != "OPENROUTER_KEY"})
        job_id = result.stdout.strip().split(";")[0]
        if not job_id.isdigit():
            raise RuntimeError(f"Invalid Slurm submission response: {result.stdout}")
        save_json(job_file, {"job_id": job_id, "stage": stage, "code": str(script.parent.parent),
                             "gpus": gpus})

    print(f"{stage}: Slurm job {job_id}", flush=True)
    wait_job(job_id, marker)


def wait_job(job_id, marker):
    """Poll Slurm until the job ends; success requires exit 0:0 and its ready marker."""
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


def read_manifest(path):
    """Saved run settings; older manifests predate offsets, token caps and shuffling."""
    if not path.exists():
        return None
    return {"start_offset": 0, "max_tokens": 1024, "previous": None, "shuffle": False,
            **json.loads(path.read_text())}


def pipeline(root, run, config=None, round_size=None, gpus=1):
    """Start a run, or resume it from its saved rounds; only the round size of new rounds may change."""
    from multi_llm.judge import judge_collection
    from core.training import TrainingConfig
    from core.workflow import save_json

    root = root.resolve()
    if gpus not in GPU_CHOICES:
        raise ValueError(f"GPUs must be one of {GPU_CHOICES}")
    if round_size is not None and round_size < 1:
        raise ValueError("Round size must be positive")

    directory = run_directory(root, run)
    directory.mkdir(parents=True, exist_ok=True)

    with (directory / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        path = directory / "experiment.json"
        manifest = read_manifest(path)
        if manifest is None:
            manifest = {"training": asdict(config or TrainingConfig()), "round_size": round_size or 128,
                        "start_offset": 0, "max_tokens": MAX_TOKENS, "previous": None, "shuffle": True}
        elif config is not None and asdict(config) != manifest["training"]:
            raise ValueError("Existing run uses a different training configuration")
        manifest["round_size"] = round_size or manifest["round_size"]
        config = TrainingConfig(**manifest["training"])

        dataset = json.loads((root / "data" / "train.json").read_text())
        if manifest["shuffle"]:
            random.Random(config.seed).shuffle(dataset)

        if not os.environ.get("OPENROUTER_KEY"):
            key_file = Path.home() / ".config" / "sysdpo" / "openrouter.json"
            os.environ["OPENROUTER_KEY"] = json.loads(key_file.read_text())["OPENROUTER_KEY"]

        save_json(path, manifest)
        offset = manifest["start_offset"]
        previous = Path(manifest["previous"]) if manifest["previous"] else None
        for index in range(len(dataset)):
            step = json.loads((previous / "state.json").read_text())["optimizer_steps"] if previous else 0
            if step >= config.max_steps:
                break
            current = directory / f"round_{index:03d}"
            if (current / "rows.json").exists():
                # A saved round keeps its rows, so an interrupted round resumes unchanged.
                rows = json.loads((current / "rows.json").read_text())
                if [row["id"] for row in rows] != [row["id"] for row in dataset[offset:offset + len(rows)]]:
                    raise ValueError(f"{current.name} rows do not match the training split order")
            else:
                rows = dataset[offset:offset + manifest["round_size"]]
                if not rows:
                    break
                current.mkdir(exist_ok=True)
                save_json(current / "rows.json", rows)
            save_json(current / "training_config.json", asdict(config))

            run_job("collect", root, current, previous, manifest["max_tokens"], gpus)
            if not (current / "preferences.ready").exists():
                judge_collection(current, JUDGE_WORKERS)

            prepare_training(current, config)
            run_job("train", root, current, previous, manifest["max_tokens"], gpus)
            previous = latest_checkpoint(current)
            if previous is None:
                raise RuntimeError("Completed training has no durable checkpoint")
            offset += len(rows)

        print(f"Experiment completed: {previous}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", required=True, help="Run directory under runs/; rerun to resume it")
    parser.add_argument("--gpus", type=int, choices=GPU_CHOICES, default=1, help="GPUs per Slurm job")
    parser.add_argument("--round-size", type=int,
                        help="Questions per new collection round; defaults to the run's saved size, else 128")
    parser.add_argument("--config", type=Path, help="Training configuration JSON for a new run")
    parser.add_argument("--status", action="store_true", help="Read saved progress and Slurm state without launching work")

    args = parser.parse_args(argv)
    if args.status:
        from multi_llm.status import show_status
        show_status(args.root.resolve(), args.run)
        return

    from core.training import TrainingConfig
    config = TrainingConfig(**json.loads(args.config.read_text())) if args.config else None
    pipeline(args.root, args.run, config, args.round_size, args.gpus)


if __name__ == "__main__":
    main()
