"""Pipeline selection, checkpoint continuation and read-only reporting."""

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

        def fake_job(stage, root, current, previous, max_tokens):
            self.jobs.append((stage, current, previous, max_tokens))
            if stage == "train":
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

    def source_run(self, start=2, count=4, legacy=False):
        directory = self.root / "runs/source"
        current = directory / "round_000"
        checkpoint = current / "checkpoints/rows_000002"
        checkpoint.mkdir(parents=True)
        rows = self.rows[start:start + count]
        manifest = {"row_ids": [row["id"] for row in rows],
                    "training": asdict(self.config), "round_size": 1024}
        if not legacy:
            manifest["start_offset"] = start
        save_json(directory / "experiment.json", manifest)
        save_json(current / "rows.json", rows)
        save_json(checkpoint / "state.json", {
            "rows_done": 2, "total_rows": 2, "optimizer_steps": 12,
            "config": asdict(self.config),
        })
        (checkpoint / "complete").touch()
        (current / "training.ready").touch()
        return checkpoint

    def test_continuation_inherits_config_offset_and_pins_checkpoint(self):
        checkpoint = self.source_run()
        pending = self.root / "runs/source/round_001"
        partial = pending / "checkpoints/rows_000001"
        partial.mkdir(parents=True)
        save_json(pending / "rows.json", self.rows[6:8])
        save_json(partial / "state.json", {
            "rows_done": 1, "total_rows": 2, "optimizer_steps": 13,
            "config": asdict(self.config),
        })
        (partial / "complete").touch()
        path = self.root / "runs/source/experiment.json"
        parent = json.loads(path.read_text())
        parent["row_ids"] += [row["id"] for row in self.rows[6:8]]
        save_json(path, parent)

        main(["--root", str(self.root), "--run", "child", "--from-run", "source",
              "--questions", "3", "--round-size", "2", "--max-tokens", "256"])
        manifest = json.loads((self.root / "runs/child/experiment.json").read_text())
        self.assertEqual(manifest["row_ids"], [16, 17, 18])
        self.assertEqual(manifest["start_offset"], 6)
        self.assertEqual(manifest["training"], asdict(self.config))
        self.assertEqual(manifest["previous"], str(checkpoint))
        self.assertEqual(self.jobs[0][2], checkpoint)
        self.assertTrue(all(job[3] == 256 for job in self.jobs))

        pipeline(self.root, "grandchild", 2, 8, from_run="child")
        grandchild = json.loads((self.root / "runs/grandchild/experiment.json").read_text())
        self.assertEqual(grandchild["start_offset"], 9)
        self.assertEqual(grandchild["row_ids"], [19, 20])

        state = json.loads((partial / "state.json").read_text())
        state["rows_done"] = 2
        save_json(partial / "state.json", state)
        (pending / "training.ready").touch()
        pipeline(self.root, "child", 3, 8, round_size=2, from_run="source", max_tokens=256)
        self.assertEqual(json.loads((self.root / "runs/child/experiment.json").read_text()), manifest)

    def test_all_selects_remaining_rows_and_offset_can_be_overridden(self):
        self.source_run()
        pipeline(self.root, "remaining", "all", 8, from_run="source")
        selected = json.loads((self.root / "runs/remaining/experiment.json").read_text())
        self.assertEqual(selected["row_ids"], [16, 17, 18, 19, 20, 21])

        pipeline(self.root, "override", 2, 8, from_run="source", start_offset=9)
        selected = json.loads((self.root / "runs/override/experiment.json").read_text())
        self.assertEqual(selected["row_ids"], [19, 20])

        main(["--root", str(self.root), "--run", "offset_only", "--start-offset", "10",
              "--questions", "all"])
        selected = json.loads((self.root / "runs/offset_only/experiment.json").read_text())
        shuffled = list(self.rows)
        random.Random(TrainingConfig().seed).shuffle(shuffled)
        self.assertEqual(selected["row_ids"], [row["id"] for row in shuffled[10:]])

    def test_legacy_manifest_can_resume_and_be_used_as_source(self):
        self.source_run(start=0, legacy=True)
        pipeline(self.root, "child", 2, 8, from_run="source")
        selected = json.loads((self.root / "runs/child/experiment.json").read_text())
        self.assertEqual(selected["start_offset"], 4)
        pipeline(self.root, "source", 4, 8, self.config, round_size=1024, max_tokens=1024)

    def test_shuffle_is_stable_for_resume_and_continuation(self):
        pipeline(self.root, "shuffled", 4, 8, self.config, round_size=2)
        path = self.root / "runs/shuffled/experiment.json"
        manifest = json.loads(path.read_text())
        shuffled = list(self.rows)
        random.Random(self.config.seed).shuffle(shuffled)
        self.assertEqual(manifest["row_ids"], [row["id"] for row in shuffled[:4]])
        pipeline(self.root, "shuffled", 4, 8, self.config, round_size=2)
        self.assertEqual(json.loads(path.read_text()), manifest)

        pipeline(self.root, "next", 2, 8, from_run="shuffled")
        child = json.loads((self.root / "runs/next/experiment.json").read_text())
        self.assertEqual(child["row_ids"], [row["id"] for row in shuffled[4:6]])

    def test_cap_stops_new_collection_and_pins_partial_terminal_checkpoint(self):
        config = TrainingConfig(max_steps=2)
        pipeline(self.root, "capped", "all", 8, config, round_size=2)
        directory = self.root / "runs/capped"
        self.assertEqual(len(list(directory.glob("round_*"))), 2)
        self.assertEqual(len(self.jobs), 4)
        terminal = directory / "round_001/checkpoints/rows_000001"
        state = json.loads((terminal / "state.json").read_text())
        state.update(rows_done=1, stop_reason="max_steps")
        save_json(terminal / "state.json", state)
        self.jobs.clear()
        pipeline(self.root, "after_cap", "all", 8, from_run="capped")
        child = json.loads((self.root / "runs/after_cap/experiment.json").read_text())
        self.assertEqual(child["previous"], str(terminal))
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

        def job(stage, root, current, previous, max_tokens):
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
            pipeline(self.root, "aligned", "all", 8, config, round_size=4)
            self.assertEqual(events, [(4, 0), (4, 1), (4, 1)])
            pipeline(self.root, "aligned", "all", 8, config, round_size=4)
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

    def test_invalid_selections_and_changed_configuration_submit_nothing(self):
        self.source_run()
        for options in (
            {"start_offset": -1}, {"start_offset": len(self.rows)}, {"questions": 0},
            {"max_tokens": 0}, {"from_run": "source", "config": TrainingConfig()},
            {"from_run": "bad/name"},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                pipeline(self.root, "invalid", options.get("questions", 2), 8,
                         **{key: value for key, value in options.items() if key != "questions"})
        self.assertEqual(self.jobs, [])

        pipeline(self.root, "fixed", 2, 8, max_tokens=512)
        self.jobs.clear()
        with self.assertRaisesRegex(ValueError, "different configuration"):
            pipeline(self.root, "fixed", 2, 8, max_tokens=256)
        self.assertEqual(self.jobs, [])

    def test_source_must_be_complete_and_match_dataset(self):
        checkpoint = self.source_run()
        pending = checkpoint.parent / "rows_000003.partial"
        pending.mkdir()
        (pending / "complete").touch()
        (checkpoint / "complete").unlink()
        with self.assertRaisesRegex(ValueError, "no completed training round"):
            pipeline(self.root, "child", 2, 8, from_run="source")
        (checkpoint / "complete").touch()
        save_json(self.root / "data/train.json", list(reversed(self.rows)))
        with self.assertRaisesRegex(ValueError, "do not match"):
            pipeline(self.root, "child", 2, 8, from_run="source")
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
