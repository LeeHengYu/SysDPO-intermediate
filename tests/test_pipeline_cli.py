"""Pipeline selection, in-place resume and read-only reporting."""

from contextlib import redirect_stdout
from dataclasses import asdict
from io import StringIO
import json
import os
from pathlib import Path
import random
from subprocess import CompletedProcess
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Barrier, Event
import unittest
from unittest.mock import patch

from core.training import TrainingConfig
from core.workflow import save_json
from multi_llm.pipeline import main, pipeline
from multi_llm.status import read_round_status
from multi_llm.runs import prepare_training


class PipelineCliTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="pipeline_test_", dir=Path(__file__).parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        (self.root / "data").mkdir()
        self.rows = [{"id": i} for i in range(10, 22)]
        save_json(self.root / "data/train.json", self.rows)
        self.config = TrainingConfig(warmup_steps=3, seed=7)
        self.jobs = []
        self.fail_train = set()

        def fake_job(stage, root, current, previous, max_tokens, gpus):
            # Like run_job, a stage with its ready marker is skipped.
            if (current / ("training.ready" if stage == "train" else "candidates.ready")).exists():
                return
            self.jobs.append((stage, current, previous, max_tokens, gpus))
            if stage == "train":
                if current.name in self.fail_train:
                    raise RuntimeError("train job failed")
                selected = json.loads((current / "rows.json").read_text())
                config = json.loads((current / "training_config.json").read_text())
                prior_step = json.loads((previous / "state.json").read_text())["optimizer_steps"] if previous else 0
                checkpoint = current / "checkpoints/rows_000001"
                checkpoint.mkdir(parents=True, exist_ok=True)
                save_json(checkpoint / "state.json", {
                    "rows_done": len(selected), "total_rows": len(selected),
                    "optimizer_steps": prior_step + 1, "config": config,
                })
                (checkpoint / "complete").touch()
            (current / ("training.ready" if stage == "train" else "candidates.ready")).touch()

        for mocked in (
            patch.dict(os.environ, {"OPENROUTER_KEY": "test-only"}),
            patch("multi_llm.pipeline.run_job", side_effect=fake_job),
            patch("multi_llm.pipeline.prepare_training"),
            patch("multi_llm.judge.judge_collection",
                  side_effect=lambda current, workers: (current / "preferences.ready").touch()),
        ):
            mocked.start()
            self.addCleanup(mocked.stop)

    def collected(self, run):
        directory = self.root / "runs" / run
        return [json.loads(path.read_text()) for path in sorted(directory.glob("round_*/rows.json"))]

    def source_run(self, start=2, count=4):
        directory = self.root / "runs/source"
        current = directory / "round_000"
        checkpoint = current / "checkpoints/rows_000002"
        checkpoint.mkdir(parents=True)
        rows = self.rows[start:start + count]
        save_json(directory / "experiment.json", {"training": asdict(self.config), "round_size": 1024,
                                                  "start_offset": start})
        save_json(current / "rows.json", rows)
        save_json(checkpoint / "state.json", {
            "rows_done": 2, "total_rows": 2, "optimizer_steps": 12,
            "config": asdict(self.config),
        })
        (checkpoint / "complete").touch()
        (current / "training.ready").touch()
        return checkpoint

    def test_rerun_resumes_interrupted_round_and_resizes_only_new_rounds(self):
        self.fail_train.add("round_001")
        with self.assertRaisesRegex(RuntimeError, "train job failed"):
            main(["--root", str(self.root), "--run", "resumed", "--round-size", "4", "--gpus", "2"])
        self.fail_train.clear()
        self.jobs.clear()

        main(["--root", str(self.root), "--run", "resumed", "--round-size", "2", "--gpus", "4"])
        groups = self.collected("resumed")
        self.assertEqual([len(group) for group in groups], [4, 4, 2, 2])
        shuffled = list(self.rows)
        random.Random(TrainingConfig().seed).shuffle(shuffled)
        self.assertEqual([row for group in groups for row in group], shuffled)
        self.assertEqual(self.jobs[0][:2], ("train", self.root / "runs/resumed/round_001"))
        self.assertEqual(self.jobs[0][2], self.root / "runs/resumed/round_000/checkpoints/rows_000001")
        self.assertTrue(all(job[3] == 512 and job[4] == 4 for job in self.jobs))
        manifest = json.loads((self.root / "runs/resumed/experiment.json").read_text())
        self.assertEqual(manifest["round_size"], 2)

        self.jobs.clear()
        pipeline(self.root, "resumed")
        self.assertEqual(self.jobs, [])
        self.assertEqual(len(self.collected("resumed")), 4)

    def test_legacy_continuation_manifest_resumes_from_its_pinned_checkpoint(self):
        checkpoint = self.source_run()
        directory = self.root / "runs/child"
        directory.mkdir()
        save_json(directory / "experiment.json", {
            "row_ids": [row["id"] for row in self.rows[6:]], "training": asdict(self.config),
            "round_size": 3, "start_offset": 6, "max_tokens": 256, "from_run": "source",
            "previous": str(checkpoint), "shuffle": False})
        pipeline(self.root, "child")
        self.assertEqual(self.collected("child"), [self.rows[6:9], self.rows[9:12]])
        self.assertEqual(self.jobs[0][2], checkpoint)
        self.assertTrue(all(job[3] == 256 for job in self.jobs))
        manifest = json.loads((directory / "experiment.json").read_text())
        self.assertEqual(manifest["from_run"], "source")

    def test_oldest_manifest_keeps_its_token_cap_and_unshuffled_order(self):
        directory = self.root / "runs/old"
        directory.mkdir(parents=True)
        save_json(directory / "experiment.json", {"training": asdict(self.config), "round_size": 6})
        pipeline(self.root, "old")
        self.assertEqual(self.collected("old"), [self.rows[:6], self.rows[6:]])
        self.assertTrue(all(job[3] == 1024 for job in self.jobs))

    def test_cap_stops_new_collection_and_rerun_submits_nothing(self):
        config = TrainingConfig(max_steps=2)
        pipeline(self.root, "capped", config, round_size=2)
        directory = self.root / "runs/capped"
        self.assertEqual(len(list(directory.glob("round_*"))), 2)
        self.assertEqual(len(self.jobs), 4)
        self.jobs.clear()
        pipeline(self.root, "capped", round_size=3)
        self.assertEqual(self.jobs, [])

    def test_fixed_rounds_do_not_refill_or_overlap(self):
        config = TrainingConfig(optimizer_rows=4, max_steps=2)
        events = []

        def judge(current, workers):
            selected = json.loads((current / "rows.json").read_text())
            usable = selected[1:] if current.name == "round_000" else [] if current.name == "round_001" else selected
            save_json(current / "scoring_requests.json", usable)
            save_json(current / "intermediate_preferences.json", [])
            (current / "preferences.ready").touch()

        def job(stage, root, current, previous, max_tokens, gpus):
            marker = current / ("candidates.ready" if stage == "collect" else "training.ready")
            if marker.exists():
                return
            prior = json.loads((previous / "state.json").read_text())["optimizer_steps"] if previous else 0
            selected = json.loads((current / "rows.json").read_text())
            if stage == "collect":
                events.append((len(selected), prior))
            else:
                inputs = json.loads((current / "training/scoring_requests.json").read_text())
                self.assertLessEqual(len(inputs), config.optimizer_rows)
                done = len(inputs)
                checkpoint = current / f"checkpoints/rows_{done:06d}"
                checkpoint.mkdir(parents=True)
                save_json(checkpoint / "state.json", {
                    "rows_done": done, "total_rows": len(inputs),
                    "optimizer_steps": prior + bool(done),
                    "config": asdict(config),
                    "stop_reason": "collection_complete"})
                (checkpoint / "complete").touch()
            marker.touch()

        with patch("multi_llm.pipeline.run_job", side_effect=job), \
                patch("multi_llm.pipeline.prepare_training", side_effect=prepare_training), \
                patch("multi_llm.judge.judge_collection", side_effect=judge):
            pipeline(self.root, "aligned", config, round_size=4)
            self.assertEqual(events, [(4, 0), (4, 1), (4, 1)])
            pipeline(self.root, "aligned", config, round_size=4)
            self.assertEqual(events, [(4, 0), (4, 1), (4, 1)])
        directory = self.root / "runs/aligned"
        self.assertEqual(len(list(directory.glob("round_*"))), 3)
        collected = [json.loads((directory / f"round_{i:03d}/rows.json").read_text()) for i in range(3)]
        ids = [row["id"] for group in collected for row in group]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {row["id"] for row in self.rows})
        self.assertEqual(json.loads((directory / "round_001/training/scoring_requests.json").read_text()), [])
        third = json.loads((directory / "round_002/training/scoring_requests.json").read_text())
        self.assertEqual(third, collected[2])

    def test_training_row_sample_is_saved_with_only_its_preferences(self):
        current = self.root / "round_000"
        current.mkdir()
        requests = [{"id": i} for i in range(128)]
        preferences = [{"id": i, "pair": j} for i in range(128) for j in range(3)]
        save_json(current / "scoring_requests.json", requests)
        save_json(current / "intermediate_preferences.json", preferences)
        config = TrainingConfig()
        prepare_training(current, config)
        expected = random.Random(f"{config.seed}:{current.name}").sample(requests, 64)
        selected = json.loads((current / "training/scoring_requests.json").read_text())
        self.assertEqual(selected, expected)
        pairs = json.loads((current / "training/intermediate_preferences.json").read_text())
        self.assertEqual(len(pairs), 192)
        self.assertEqual({pair["id"] for pair in pairs}, {row["id"] for row in selected})
        selection = json.loads((current / "training/selection.json").read_text())
        self.assertEqual(selection["usable_rows"], 128)
        self.assertEqual(selection["selected_row_ids"], [row["id"] for row in selected])
        (current / "training/inputs.ready").unlink()
        prepare_training(current, config)
        self.assertEqual(json.loads((current / "training/scoring_requests.json").read_text()), selected)

    def test_invalid_settings_and_changed_data_submit_nothing(self):
        for options in ({"gpus": 3}, {"round_size": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                pipeline(self.root, "invalid", **options)
        self.assertFalse((self.root / "runs/invalid").exists())

        pipeline(self.root, "fixed", self.config, round_size=6)
        self.jobs.clear()
        with self.assertRaisesRegex(ValueError, "different training configuration"):
            pipeline(self.root, "fixed", TrainingConfig())
        save_json(self.root / "data/train.json", list(reversed(self.rows)))
        with self.assertRaisesRegex(ValueError, "do not match"):
            pipeline(self.root, "fixed")
        self.assertEqual(self.jobs, [])

    def test_status_reports_saved_vs_logged_progress_without_writes(self):
        checkpoint = self.source_run()
        state = json.loads((checkpoint / "state.json").read_text())
        state["last_audit"] = {"L_sys": 0.6, "L_A": 0.4, "L_A_weighted": 0.2}
        save_json(checkpoint / "state.json", state)
        current = checkpoint.parent.parent
        (current / "training").mkdir()
        save_json(current / "training/selection.json", {"usable_rows": 4, "selected_row_ids": [12, 13]})
        (current.parent / "round_001").mkdir()
        for stage, job in (("collect", "100"), ("train", "101")):
            save_json(current / f"{stage}_job.json", {"job_id": job})
        save_json(current / "question_12.json", {
            "y1": "one", "y2": "two", "y3": "three",
            "finish_reasons": {"y1": "stop", "y2": "length", "y3": "stop"},
        })
        (current / "calls").mkdir()
        save_json(current / "calls/one.json", {"response": {"usage": {"cost": 0.01}}})
        save_json(current / "calls/two.json", {"error": "request started"})
        (current / "train_101.log").write_text("rows=3/4 step=13 loss=1.02 pairs=9\n")
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        output = StringIO()
        with patch("multi_llm.status.subprocess.run", return_value=CompletedProcess(
                [], 0, "100|COMPLETED|0:0|00:01:00|\n101|RUNNING|0:0|00:00:30|\n", "")) as query, \
                redirect_stdout(output):
            main(["--root", str(self.root), "--run", "source", "--status",
                  "--config", str(self.root / "does_not_exist.json")])
        text = output.getvalue()
        self.assertIn("Training-split start: 2", text)
        self.assertIn("Collected: 1/4", text)
        self.assertIn("Train: RUNNING", text)
        self.assertIn("attempted: 2 | successful: 1", text)
        self.assertIn("$0.010000", text)
        self.assertIn("rows 3/4 | cumulative step 13", text)
        self.assertIn("Durably saved: 2/2 rows | cumulative step 12", text)
        self.assertIn("L_sys 0.600000 | L_A 0.400000 | weighted L_A 0.200000", text)
        self.assertIn("Selected for update: 2/4 usable rows", text)
        self.assertIn("Cost is incomplete", text)
        self.assertIn("Preparing round metadata", text)
        self.assertEqual(query.call_args.args[0][0], "sacct")
        self.assertEqual(self.jobs, [])
        self.assertEqual({p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}, before)

    def test_status_reads_rounds_concurrently_and_prints_in_order(self):
        checkpoint = self.source_run()
        first = checkpoint.parent.parent
        second = first.parent / "round_001"
        second.mkdir()
        save_json(second / "rows.json", self.rows[6:8])
        for current, cost in ((first, 0.01), (second, 0.02)):
            (current / "calls").mkdir()
            save_json(current / "calls/one.json", {"response": {"usage": {"cost": cost}}})

        started = Barrier(2)
        second_finished = Event()

        def read_concurrently(current, *args):
            started.wait(timeout=5)
            report = read_round_status(current, *args)
            if current == first:
                self.assertTrue(second_finished.wait(timeout=5))
            else:
                second_finished.set()
            return report

        output = StringIO()
        with patch("multi_llm.status.read_round_status", side_effect=read_concurrently), redirect_stdout(output):
            main(["--root", str(self.root), "--run", "source", "--status"])

        text = output.getvalue()
        self.assertLess(text.index("\nround_000\n"), text.index("\nround_001\n"))
        self.assertIn("Run API totals: attempted 2 | successful 2 | recorded cost $0.030000", text)

    def test_status_starts_without_torch_api_credentials_or_slurm(self):
        checkpoint = self.source_run()
        save_json(checkpoint.parent.parent / "collect_job.json", {"job_id": "100"})
        env = {key: value for key, value in os.environ.items() if key != "OPENROUTER_KEY"}
        env["PATH"] = ""
        result = subprocess.run([
            sys.executable, "-c",
            "import sys; from multi_llm.pipeline import main; main(sys.argv[1:]); "
            "assert 'torch' not in sys.modules; assert 'openrouter' not in sys.modules",
            "--root", str(self.root), "--run", "source", "--status",
        ], check=True, capture_output=True, text=True, env=env)
        self.assertIn("Durably saved: 2/2", result.stdout)
        self.assertIn("Slurm unavailable", result.stdout)


if __name__ == "__main__":
    unittest.main()
