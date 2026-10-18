"""Read saved run locations, checkpoints and dataset positions."""

import json
from pathlib import Path


def run_directory(root, run):
    if not run or Path(run).name != run or run in (".", ".."):
        raise ValueError("Run must be a single directory name")
    return root / "runs" / run


def latest_checkpoint(directory):
    completed = sorted(path for path in (directory / "checkpoints").glob("rows_*/complete")
                       if path.parent.name.removeprefix("rows_").isdigit())
    return completed[-1].parent if completed else None


def continuation(root, run, dataset, checkpoint=None):
    directory = run_directory(root, run)
    manifest = json.loads((directory / "experiment.json").read_text())
    rounds = sorted(directory.glob("round_*"))

    if checkpoint is None:
        for current in reversed(rounds):
            candidate = latest_checkpoint(current)
            if candidate and (current / "training.ready").exists():
                state = json.loads((candidate / "state.json").read_text())
                if state["rows_done"] == state["total_rows"]:
                    checkpoint = candidate
                    break
        else:
            raise ValueError("Source run has no completed training round")

    checkpoint = Path(checkpoint)
    current = checkpoint.parent.parent
    state = json.loads((checkpoint / "state.json").read_text())
    if (current.parent != directory or not (checkpoint / "complete").exists()
            or not (current / "training.ready").exists() or state["rows_done"] != state["total_rows"]):
        raise ValueError("Source checkpoint is not a completed training round")

    consumed = [row["id"] for item in rounds if item <= current
                for row in json.loads((item / "rows.json").read_text())]
    start = manifest.get("start_offset", 0)
    end = start + len(consumed)
    if ([row["id"] for row in dataset[start:end]] != consumed
            or manifest["row_ids"][:len(consumed)] != consumed):
        raise ValueError("Source run rows do not match the current training split")

    return checkpoint, end, state["config"]
