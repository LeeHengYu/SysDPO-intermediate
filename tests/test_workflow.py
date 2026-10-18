"""Offline checks for concurrent judging and durable stage handoff."""
from pathlib import Path
from datetime import timedelta
from itertools import permutations, product
import json
import os
from subprocess import CompletedProcess
import shutil
import threading
import time
import unittest
from unittest.mock import patch
import uuid

from multi_llm.judge import judge_collection
from multi_llm.collect import collect
from multi_llm.pipeline import pipeline, run_job
from core.training import TrainingConfig
from core.schemas import DatasetRow, Judgment, QuestionRecord
from core.distributed import distributed_job
from core.topo_sort import topological_sort
from core.workflow import PROMPT_FILE, load_prompts, model_messages, save_json


class WorkflowTests(unittest.TestCase):
    def test_three_candidate_graph_accepts_exactly_acyclic_preferences(self):
        names = ("y1", "y2", "y3")
        for directions in product((False, True), repeat=3):
            edges = [(b, a) if reverse else (a, b)
                     for (a, b), reverse in zip((("y1", "y2"), ("y1", "y3"), ("y2", "y3")), directions)]
            expected = any(all(order.index(a) < order.index(b) for a, b in edges)
                           for order in permutations(names))
            with self.subTest(edges=edges):
                self.assertEqual(topological_sort(edges), expected)
        self.assertTrue(topological_sort([]))
        self.assertFalse(topological_sort([("y1", "y1")]))

    def test_configurable_collection_rounds_and_checkpoint_handoff(self):
        root = Path(__file__).parent / ("round_test_" + uuid.uuid4().hex)
        (root / "data").mkdir(parents=True)
        rows = [{"id": i} for i in range(1030)]
        save_json(root / "data" / "train.json", rows)

        try:
            for size in (512, 1025):
                with self.subTest(round_size=size):
                    calls = []

                    def fake_job(stage, project, directory, previous, max_tokens):
                        calls.append((stage, directory, previous))
                        if stage == "train":
                            checkpoint = directory / "checkpoints" / "rows_000001"
                            checkpoint.mkdir(parents=True)
                            (checkpoint / "complete").touch()

                    with patch.dict(os.environ, {"OPENROUTER_KEY": "test-only"}), \
                            patch("multi_llm.pipeline.run_job", side_effect=fake_job), \
                            patch("multi_llm.judge.judge_collection") as judge:
                        pipeline(root, f"rounds{size}", len(rows), 8, TrainingConfig(), size)

                    directory = root / "runs" / f"rounds{size}"
                    rounds = sorted(directory.glob("round_*"))
                    selected = [json.loads((item / "rows.json").read_text()) for item in rounds]
                    self.assertEqual([row for batch in selected for row in batch], rows)
                    self.assertEqual([len(batch) for batch in selected],
                                     [len(rows[offset:offset + size]) for offset in range(0, len(rows), size)])
                    self.assertEqual(judge.call_count, len(rounds))
                    self.assertEqual(json.loads((directory / "experiment.json").read_text())["round_size"], size)
                    for index, item in enumerate(rounds):
                        previous = rounds[index - 1] / "checkpoints" / "rows_000001" if index else None
                        self.assertEqual(calls[2 * index:2 * index + 2],
                                         [("collect", item, previous), ("train", item, previous)])

                    with patch.dict(os.environ, {"OPENROUTER_KEY": "test-only"}):
                        with self.assertRaisesRegex(ValueError, "different configuration"):
                            pipeline(root, f"rounds{size}", len(rows), 8, TrainingConfig(), size + 1)

            with self.assertRaisesRegex(ValueError, "Round size must be positive"):
                pipeline(root, "invalid", len(rows), 8, TrainingConfig(), 0)
        finally:
            shutil.rmtree(root)

    def test_collection_default_has_no_1024_row_limit(self):
        root = Path(__file__).parent / ("collection_test_" + uuid.uuid4().hex)
        (root / "base").mkdir(parents=True)
        save_json(root / "base" / "provenance.json", {"model_id": "test"})
        rows = [DatasetRow(id=i, question="question", chosen="yes", rejected="no") for i in range(1030)]
        save_json(root / "rows.json", rows)

        try:
            for count, expected in ((None, 1030), (7, 7)):
                publish = root / f"publish_{count}"
                with self.subTest(count=count), \
                        patch("multi_llm.collect.AutoTokenizer.from_pretrained",
                              side_effect=RuntimeError("stop before model loading")):
                    with self.assertRaisesRegex(RuntimeError, "stop before model loading"):
                        collect(root / "base", root / "rows.json", root / "output", publish,
                                count=count, device="cpu")
                self.assertEqual(len(json.loads((publish / "dataset.json").read_text())["rows"]), expected)
        finally:
            shutil.rmtree(root)

    def test_distributed_job_allows_long_shards_and_closes_on_failure(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "2", "LOCAL_RANK": "1"}), \
                patch("core.distributed.torch.cuda.set_device") as select, \
                patch("core.distributed.dist.init_process_group") as initialize, \
                patch("core.distributed.dist.destroy_process_group") as close:
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                with distributed_job():
                    raise RuntimeError("worker failed")

            select.assert_called_once_with(1)
            initialize.assert_called_once_with(backend="nccl", timeout=timedelta(hours=30))
            close.assert_called_once_with()

    def test_submission_omits_api_key_and_reuses_completed_stage(self):
        root = Path(__file__).parent / ("submission_test_" + uuid.uuid4().hex)
        root.mkdir()
        try:
            def command(args, **kwargs):
                if args[0] == "sbatch":
                    script = Path(__file__).resolve().parents[1] / "script" / "job.sh"
                    self.assertEqual(args[args.index("--output") + 2], str(script))
                    self.assertTrue(script.is_file())
                    self.assertEqual(args[-2], str(script.parent.parent))
                    self.assertEqual(args[-1], "512")
                    self.assertNotIn("OPENROUTER_KEY", kwargs["env"])
                    self.assertIsNone(json.loads((root / f"{stage}_job.json").read_text())["job_id"])
                    self.assertEqual(args[args.index("--time") + 1],
                                     "30:00:00" if stage == "collect" else "04:00:00")
                    return CompletedProcess(args, 0, "123\n", "")
                (root / ("training.ready" if stage == "train" else "candidates.ready")).touch()
                return CompletedProcess(args, 0, "COMPLETED|0:0|\n", "")

            with patch.dict(os.environ, {"OPENROUTER_KEY": "test-only"}), \
                    patch("multi_llm.pipeline.subprocess.run", side_effect=command) as submitted:
                for stage in ("train", "collect"):
                    run_job(stage, root, root, max_tokens=512)
                    run_job(stage, root, root, max_tokens=512)
            self.assertEqual(submitted.call_count, 4)
            self.assertEqual(json.loads((root / "train_job.json").read_text())["job_id"], "123")
        finally:
            shutil.rmtree(root)

    def test_parallel_pairs_cycles_and_duplicate_prefilter(self):
        root = Path(__file__).parent / ("judge_test_" + uuid.uuid4().hex)
        root.mkdir()
        lock = threading.Lock()
        active, peak, calls = 0, 0, []
        try:
            prompts = load_prompts(PROMPT_FILE)
            save_json(root / "collection_config.json", {"prompts": prompts})
            rows = [DatasetRow(id=i, question="question", chosen="yes", rejected="no") for i in range(3)]
            save_json(root / "dataset.json", {"rows": rows})
            for row in rows:
                record = QuestionRecord(id=row.id, x=row.question, system="", z_chosen="yes", z_rejected="no",
                    a_messages=model_messages(row, prompts, prompts.a_instruction),
                    y1="one", y2="two", y3="three" if row.id < 2 else " one ",
                    finish_reasons={"y1": "stop", "y2": "stop", "y3": "stop"})
                save_json(root / f"question_{row.id}.json", record)
            (root / "candidates.ready").touch()

            def fake_judge(directory, row, first, second, stage, first_id, second_id, prompts):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                    calls.append(row.id)
                time.sleep(0.02)
                winner = second_id if row.id == 1 and first_id == "y1" and second_id == "y3" else first_id
                with lock:
                    active -= 1
                return Judgment(choice="option1", presented_order=[first_id, second_id], winner=winner)

            with patch("multi_llm.judge.judge_pair", side_effect=fake_judge):
                judge_collection(root, workers=3)
            self.assertGreater(peak, 1)
            self.assertLessEqual(peak, 3)
            self.assertEqual(sorted(calls), [0, 0, 0, 1, 1, 1])
            preferences = json.loads((root / "intermediate_preferences.json").read_text())
            self.assertEqual([p["id"] for p in preferences], [0, 0, 0])
            requests = json.loads((root / "scoring_requests.json").read_text())
            self.assertEqual(len(requests), 3)
            self.assertEqual(requests[2]["candidate_mask"], [True, True, False])
            self.assertTrue((root / "preferences.ready").exists())
        finally:
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
