"""Read-only run reporting; no model or API dependencies."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from decimal import Decimal
import json
import re
import subprocess

from multi_llm.runs import latest_checkpoint, run_directory


@dataclass
class RoundReport:
    lines: list[str] = field(default_factory=list)
    attempted: int = 0
    successful: int = 0
    missing_cost: int = 0
    cost: Decimal = Decimal("0")


def read_round_status(current, jobs, states, optimizer_rows):
    report = RoundReport(lines=[f"\n{current.name}"])
    if not (current / "rows.json").exists():
        report.lines.append("Preparing round metadata.")
        return report

    selected = json.loads((current / "rows.json").read_text())
    records = [json.loads(path.read_text()) for path in current.glob("question_*.json")]
    incomplete = sum(any(row["finish_reasons"].get(y) != "stop" or not row[y].strip()
                         for y in ("y1", "y2", "y3")) for row in records)

    for stage, marker in (("collect", "candidates.ready"), ("train", "training.ready")):
        job = jobs.get(current / f"{stage}_job.json")
        job_id = str(job["job_id"]) if job and job["job_id"] is not None else None
        state = states.get(job_id, "UNKNOWN" if job_id else "SUBMITTING" if job else "NOT SUBMITTED")
        ready = (current / marker).exists()
        report.lines.append(f"{stage.capitalize()}: {state} | job {job_id or '-'} | output ready: {ready}")
    report.lines.append(f"Collected: {len(records)}/{len(selected)} questions | incomplete triples: {incomplete}")

    calls = [json.loads(path.read_text()) for path in (current / "calls").glob("*.json")]
    report.attempted = len(calls)
    report.successful = sum("response" in call and "error" not in call for call in calls)
    costs = [Decimal(str(cost)) for call in calls
             if (cost := call.get("response", {}).get("usage", {}).get("cost")) is not None]
    report.missing_cost = len(calls) - len(costs)
    report.cost = sum(costs, Decimal("0"))
    judge_state = "COMPLETED" if (current / "preferences.ready").exists() else "IN PROGRESS" if calls else "NOT STARTED"
    report.lines.append(f"Judging: {judge_state} | attempted: {report.attempted} | successful: {report.successful}")
    report.lines.append(f"Recorded API cost: ${report.cost:.6f} | calls without recorded cost: {report.missing_cost}")

    summary_path = current / "judging_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        outcomes = Counter(json.loads(path.read_text()).get("discard_reason") or "accepted"
                           for path in current.glob("judged_*.json"))
        report.lines.append(f"Training data: {summary['system_rows']} system rows | {summary['accepted_pairs']} intermediate pairs")
        report.lines.append(f"Filtering: {dict(outcomes)} | judge errors: {len(summary['errors'])}")

    training_job = jobs.get(current / "train_job.json")
    log = current / f"train_{training_job['job_id']}.log" if training_job else None
    if log and log.exists():
        batches = re.findall(r"rows=(\d+)/(\d+) step=(\d+) loss=([\d.eE+-]+) pairs=(\d+)", log.read_text())
        if batches:
            done, total, step, loss, pairs = batches[-1]
            batch_rows = int(done) - (int(batches[-2][0]) if len(batches) > 1 else 0)
            report.lines.append(f"Last logged: rows {done}/{total} | cumulative step {step} | loss {loss} | pairs {pairs}")
            if len(batches) > 1 or int(done) <= optimizer_rows:
                report.lines.append(f"Last logged batch: {batch_rows} rows")

    checkpoint = latest_checkpoint(current)
    if checkpoint:
        state = json.loads((checkpoint / "state.json").read_text())
        report.lines.append(f"Durably saved: {state['rows_done']}/{state['total_rows']} rows | cumulative step {state['optimizer_steps']}")
        report.lines.append(f"Checkpoint: {checkpoint}")

    return report


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
    with ThreadPoolExecutor(max_workers=8) as pool:
        reports = [pool.submit(read_round_status, current, jobs, states,
                               manifest["training"]["optimizer_rows"]) for current in rounds]
        for future in reports:
            report = future.result()
            print("\n".join(report.lines), flush=True)
            total_calls += report.attempted
            total_success += report.successful
            total_missing += report.missing_cost
            total_cost += report.cost

    print(f"\nRun API totals: attempted {total_calls} | successful {total_success} | recorded cost ${total_cost:.6f}")
    if total_missing:
        print(f"Cost is incomplete: {total_missing} calls have no recorded cost.")
