"""Read saved run locations and checkpoints."""

import json
from pathlib import Path
import random


def run_directory(root, run):
    if not run or Path(run).name != run or run in (".", ".."):
        raise ValueError("Run must be a single directory name")
    return root / "runs" / run


def latest_checkpoint(directory):
    completed = sorted(path for path in (directory / "checkpoints").glob("rows_*/complete")
                       if path.parent.name.removeprefix("rows_").isdigit())
    return completed[-1].parent if completed else None


def prepare_training(directory, config):
    """Persist one reproducible row sample and all of its intermediate preferences."""
    from core.workflow import save_json

    target = directory / "training"
    if (target / "inputs.ready").exists():
        return
    requests = json.loads((directory / "scoring_requests.json").read_text())
    usable_count = len(requests)
    if usable_count > config.optimizer_rows:
        requests = random.Random(f"{config.seed}:{directory.name}").sample(requests, config.optimizer_rows)
    ids = {row["id"] for row in requests}
    preferences = [pair for pair in json.loads((directory / "intermediate_preferences.json").read_text())
                   if pair["id"] in ids]
    target.mkdir(exist_ok=True)
    save_json(target / "scoring_requests.json", requests)
    save_json(target / "intermediate_preferences.json", preferences)
    save_json(target / "selection.json", {"usable_rows": usable_count,
              "selected_row_ids": [row["id"] for row in requests], "seed": config.seed})
    (target / "inputs.ready").touch()
