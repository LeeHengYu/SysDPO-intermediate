"""Pinned input preparation without downloading models or datasets."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from datasets import Dataset

from multi_llm.prepare import prepare


class PreparationTests(unittest.TestCase):
    def test_pinned_downloads_and_disjoint_split_are_reused(self):
        dataset = Dataset.from_list([
            {"question": f"question {i}", "chosen": "yes", "rejected": "no", "system": ""}
            for i in range(20)
        ])
        with TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)

            def download(*args, **kwargs):
                Path(kwargs["local_dir"]).mkdir(parents=True, exist_ok=True)

            with patch("multi_llm.prepare.HfApi") as api, \
                    patch("multi_llm.prepare.snapshot_download", side_effect=download) as snapshot, \
                    patch("multi_llm.prepare.load_dataset", return_value=dataset) as load:
                api.return_value.model_info.return_value = SimpleNamespace(sha="model-revision")
                api.return_value.dataset_info.return_value = SimpleNamespace(sha="dataset-revision")
                prepare(root)
                saved = {name: (root / f"data/{name}.json").read_bytes() for name in ("train", "test")}
                prepare(root)

            api.assert_called_once()
            load.assert_called_once_with("Intel/orca_dpo_pairs", revision="dataset-revision", split="train")
            self.assertEqual(snapshot.call_count, 2)
            self.assertEqual(snapshot.call_args.kwargs["revision"], "model-revision")
            train = json.loads(saved["train"])
            test = json.loads(saved["test"])
            self.assertEqual((len(train), len(test)), (19, 1))
            self.assertFalse({row["id"] for row in train} & {row["id"] for row in test})
            self.assertEqual({row["id"] for row in train + test}, set(range(20)))
            self.assertTrue((root / "data/complete").exists())
            for name, contents in saved.items():
                self.assertEqual((root / f"data/{name}.json").read_bytes(), contents)
