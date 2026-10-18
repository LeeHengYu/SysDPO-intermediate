"""Read-only run reporting; no model or API dependencies."""

from collections import Counter
from decimal import Decimal
import json
import re
import subprocess

from multi_llm.runs import latest_checkpoint, run_directory


def show_status(root, run):
    directory = run_directory(root, run)
    manifest = json.loads((directory / "experiment.json").read_text())
    rounds = sorted(directory.glob("round_*"))
    jobs = {path: json.loads(path.read_text()) for current in rounds
            for stage in ("collect", "train") if (path := current / f"{stage}_job.json").exists()}
    job_ids = [str(job["job_id"]) for job in jobs.values() if job["job_id"] is not None]
    states = {}
    if job_ids:
        try:
            result = subprocess.run(["sacct", "-nX", "-P", "-j", ",".join(job_ids),
                                     "--format=JobID,State,ExitCode,Elapsed"],
                                    check=True, text=True, capture_output=True)
            states = {fields[0]: " | ".join(fields[1:4]) for line in result.stdout.splitlines()
                      if len(fields := line.split("|")) >= 4}
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("Slurm unavailable; showing saved progress only.")

    print(f"Run: {run}")
    print(f"Selected questions: {len(manifest['row_ids'])} | collection round size: {manifest['round_size']}")
    print(f"Optimizer batch: {manifest['training']['optimizer_rows']} rows")
    if manifest.get("previous"):
        print(f"Starting checkpoint: {manifest['previous']}")

    total_calls, total_success, total_missing = 0, 0, 0
    total_cost = Decimal("0")
    for current in rounds:
        print(f"\n{current.name}")
        if not (current / "rows.json").exists():
            print("Preparing round metadata.")
            continue

        selected = json.loads((current / "rows.json").read_text())
        records = [json.loads(path.read_text()) for path in current.glob("question_*.json")]
        incomplete = sum(any(row["finish_reasons"].get(y) != "stop" or not row[y].strip()
                             for y in ("y1", "y2", "y3")) for row in records)

        for stage, marker in (("collect", "candidates.ready"), ("train", "training.ready")):
            job = jobs.get(current / f"{stage}_job.json")
            job_id = str(job["job_id"]) if job and job["job_id"] is not None else None
            state = states.get(job_id, "UNKNOWN" if job_id else "SUBMITTING" if job else "NOT SUBMITTED")
            ready = (current / marker).exists()
            print(f"{stage.capitalize()}: {state} | job {job_id or '-'} | output ready: {ready}")
        print(f"Collected: {len(records)}/{len(selected)} questions | incomplete triples: {incomplete}")

        calls = [json.loads(path.read_text()) for path in (current / "calls").glob("*.json")]
        successful = sum("response" in call and "error" not in call for call in calls)
        costs = [Decimal(str(cost)) for call in calls
                 if (cost := call.get("response", {}).get("usage", {}).get("cost")) is not None]
        missing = len(calls) - len(costs)
        cost = sum(costs, Decimal("0"))
        total_calls += len(calls)
        total_success += successful
        total_missing += missing
        total_cost += cost
        judge_state = "COMPLETED" if (current / "preferences.ready").exists() else "IN PROGRESS" if calls else "NOT STARTED"
        print(f"Judging: {judge_state} | attempted: {len(calls)} | successful: {successful}")
        print(f"Recorded API cost: ${cost:.6f} | calls without recorded cost: {missing}")

        summary_path = current / "judging_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            outcomes = Counter(json.loads(path.read_text()).get("discard_reason") or "accepted"
                               for path in current.glob("judged_*.json"))
            print(f"Training data: {summary['system_rows']} system rows | {summary['accepted_pairs']} intermediate pairs")
            print(f"Filtering: {dict(outcomes)} | judge errors: {len(summary['errors'])}")

        training_job = jobs.get(current / "train_job.json")
        log = current / f"train_{training_job['job_id']}.log" if training_job else None
        if log and log.exists():
            batches = re.findall(r"rows=(\d+)/(\d+) step=(\d+) loss=([\d.eE+-]+) pairs=(\d+)", log.read_text())
            if batches:
                done, total, step, loss, pairs = batches[-1]
                batch_rows = int(done) - (int(batches[-2][0]) if len(batches) > 1 else 0)
                print(f"Last logged: rows {done}/{total} | cumulative step {step} | loss {loss} | pairs {pairs}")
                if len(batches) > 1 or int(done) <= manifest["training"]["optimizer_rows"]:
                    print(f"Last logged batch: {batch_rows} rows")

        checkpoint = latest_checkpoint(current)
        if checkpoint:
            state = json.loads((checkpoint / "state.json").read_text())
            print(f"Durably saved: {state['rows_done']}/{state['total_rows']} rows | cumulative step {state['optimizer_steps']}")
            print(f"Checkpoint: {checkpoint}")

    print(f"\nRun API totals: attempted {total_calls} | successful {total_success} | recorded cost ${total_cost:.6f}")
    if total_missing:
        print(f"Cost is incomplete: {total_missing} calls have no recorded cost.")
