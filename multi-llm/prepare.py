"""Download pinned experiment inputs on the internet-connected management node."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download

from core.schemas import DatasetRow
from core.workflow import save_json


def prepare(root):
    model_id = "Qwen/Qwen1.5-1.8B-Chat"
    dataset_id = "Intel/orca_dpo_pairs"
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "inputs.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        api = HfApi()
        manifest = {"model_id": model_id, "model_revision": api.model_info(model_id).sha,
                    "dataset_id": dataset_id, "dataset_revision": api.dataset_info(dataset_id).sha,
                    "split_seed": 42, "test_fraction": 0.05}
        save_json(manifest_path, manifest)

    model = root / "models" / "Qwen1.5-1.8B-Chat"
    snapshot_download(model_id, revision=manifest["model_revision"], local_dir=model,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "LICENSE"])
    save_json(model / "provenance.json", {"model_id": model_id, "revision": manifest["model_revision"]})

    data = root / "data"
    data.mkdir(exist_ok=True)

    if not (data / "complete").exists():
        dataset = load_dataset(dataset_id, revision=manifest["dataset_revision"], split="train")
        dataset = dataset.add_column("id", list(range(len(dataset))))
        split = dataset.train_test_split(test_size=manifest["test_fraction"], seed=manifest["split_seed"])

        for name in ("train", "test"):
            rows = [asdict(DatasetRow(**row)) for row in split[name]]
            save_json(data / f"{name}.json", rows)

        save_json(data / "manifest.json", {**manifest, "total_rows": len(dataset),
                  "train_rows": len(split["train"]), "test_rows": len(split["test"])})

        (data / "complete").write_text("complete\n")

    print(json.dumps(json.loads((data / "manifest.json").read_text()), indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)

    prepare(parser.parse_args().root)
