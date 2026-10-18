"""Offline checks; never access the user's key or make network calls."""

from contextlib import redirect_stdout
import io
import hashlib
from itertools import permutations, product
import json
import os
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import Mock, patch

import httpx

from pydantic import TypeAdapter, ValidationError

from core import model_api
from smoke import train
from core import workflow
from core.likelihood import compute_saved_losses
from core.schemas import DatasetRow, IntermediatePreference, Judgment, Prompts, QuestionRecord, ScoringRequest


def completion(text):
    return {"model": "test-model", "choices": [{"message": {"content": text},
                                                 "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


class SmokeChecks(unittest.TestCase):
    def test_three_candidate_graph_and_pair_export(self):
        names = ("y1", "y2", "y3")
        for directions in product((False, True), repeat=3):
            edges = [(b, a) if reverse else (a, b)
                     for (a, b), reverse in zip((("y1", "y2"), ("y1", "y3"), ("y2", "y3")), directions)]
            expected = any(all(order.index(a) < order.index(b) for a, b in edges)
                           for order in permutations(names))
            self.assertEqual(train.topological_sort(edges), expected)
            with self.subTest(edges=edges), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                train.save_json(output / "dataset.json", {"rows": [
                    {"id": 0, "question": "x", "chosen": "c", "rejected": "r"}]})

                def fake_call(model, messages, **settings):
                    if not settings.get("json_output"):
                        return completion(f"y{api.call_count}")
                    task = json.loads(messages[1]["content"])
                    pair = (task["candidate_1"], task["candidate_2"])
                    choice = "option1" if pair in edges else "option2"
                    return completion(json.dumps({"choice": choice}))

                with patch.object(workflow, "call_gen", side_effect=fake_call) as api, \
                        patch.object(workflow, "call_judge", side_effect=lambda state, rubric, model:
                                     api(model, [{"role": "system", "content": rubric},
                                                 {"role": "user", "content": json.dumps(state)}], json_output=True)), \
                        patch("sys.argv", ["train.py", "--questions", "1", "--output-dir", str(output)]), \
                        redirect_stdout(io.StringIO()):
                    train.main()
                    self.assertEqual(api.call_count, 6)
                pairs = json.loads((output / "intermediate_preferences.json").read_text())
                self.assertEqual(len(pairs), 3 if expected else 0)
                if expected:
                    self.assertEqual([(p["y_chosen"], p["y_rejected"]) for p in pairs], edges)
                record = json.loads((output / "question_0.json").read_text())
                self.assertEqual(record["comparisons_accepted"], expected)
                self.assertEqual(len(record["scoring_request"]["intermediates"]), 3)
        self.assertTrue(train.topological_sort([]))
        self.assertFalse(train.topological_sort([("y1", "y1")]))

    def test_non_strict_comparison_stops_remaining_calls(self):
        for label in ("tie", "duplicate"):
            for stop_at in ((1,) if label == "duplicate" else (1, 2)):
                with self.subTest(label=label, stop_at=stop_at), tempfile.TemporaryDirectory() as temporary:
                    output = Path(temporary)
                    train.save_json(output / "dataset.json", {"rows": [
                        {"id": 0, "question": "x", "chosen": "c", "rejected": "r"}]})

                    def fake_call(model, messages, **settings):
                        if not settings.get("json_output"):
                            if label == "duplicate" and api.call_count == 2:
                                return completion("y1")
                            return completion(f"y{api.call_count}")
                        choice = label if api.call_count == 3 + stop_at else "option1"
                        return completion(json.dumps({"choice": choice}))

                    with patch.object(workflow, "call_gen", side_effect=fake_call) as api, \
                        patch.object(workflow, "call_judge", side_effect=lambda state, rubric, model:
                                     api(model, [{"role": "system", "content": rubric},
                                                 {"role": "user", "content": json.dumps(state)}], json_output=True)), \
                            patch("sys.argv", ["train.py", "--questions", "1", "--output-dir", str(output)]), \
                            redirect_stdout(io.StringIO()):
                        train.main()
                        expected_calls = 3 if label == "duplicate" else 3 + stop_at
                        self.assertEqual(api.call_count, expected_calls)
                        train.main()
                        self.assertEqual(api.call_count, expected_calls)
                    self.assertEqual(json.loads((output / "intermediate_preferences.json").read_text()), [])
                    self.assertEqual(len(json.loads((output / "scoring_requests.json").read_text())), 1)
                    record = json.loads((output / "question_0.json").read_text())
                    self.assertEqual(record["discard_reason"], label)
                    if label == "duplicate":
                        self.assertEqual(record["scoring_request"]["candidate_mask"], [True, False, True])

    def test_scoring_request_hash_and_shape_validation(self):
        original = {"id": 0, "a_messages": [], "intermediates": ["y1", "y2"],
                    "b_messages": [[], []], "final_answers": ["c", "r"], "candidate_mask": [True, True]}
        request = ScoringRequest(**original)
        expected = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
        self.assertEqual(request.request_sha256, expected)
        self.assertEqual(ScoringRequest(**original, request_sha256=expected), request)
        with self.assertRaises(ValidationError):
            ScoringRequest(**original, request_sha256="wrong")
        with self.assertRaises(ValidationError):
            ScoringRequest(**{**original, "intermediates": ["y1"]})
        with self.assertRaises(ValidationError):
            ScoringRequest(**{**original, "candidate_mask": [False, False]})

    def test_record_validation_and_json_compatibility(self):
        for choice in ("1", "2", "uncertain", "duplicate"):
            with self.subTest(choice=choice), self.assertRaises(ValidationError):
                Judgment(choice=choice)
        with self.assertRaises(ValidationError):
            TypeAdapter(Judgment).validate_json('{"choice":"option1","confidence":0.9}')
        with self.assertRaises(ValidationError):
            Prompts(instruction_policy=" ", a_instruction="a", b_instruction="b", judge_rubric="j")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dataset.json"
            # Old cached dictionaries are validated into records without editing the cache.
            train.save_json(path, {"rows": [{"id": 0, "question": "x", "chosen": "c", "rejected": "r"}]})
            before = path.read_bytes()
            rows = train.load_questions(path.parent, 1)
            self.assertIsInstance(rows[0], DatasetRow)
            self.assertEqual(path.read_bytes(), before)
            record = QuestionRecord(id=0, x="x", system="", z_chosen="c", z_rejected="r", a_messages=[])
            record.intermediate_judgments = [Judgment(choice="tie")]
            train.save_json(path.parent / "question.json", record)
            saved = json.loads((path.parent / "question.json").read_text())
            self.assertNotIn("y1", saved)
            self.assertNotIn("scoring_request", saved)
            self.assertIsNone(saved["intermediate_judgments"][0]["winner"])
            train.save_json(path, {"rows": [{"id": 0, "question": " ", "chosen": "c", "rejected": "r"}]})
            with self.assertRaises(ValidationError):
                train.load_questions(path.parent, 1)

    def test_entry_point(self):
        with patch("sys.argv", ["train.py", "--help"]), redirect_stdout(io.StringIO()) as terminal:
            with self.assertRaises(SystemExit) as result:
                runpy.run_module("smoke.train", run_name="__main__")
        self.assertEqual(result.exception.code, 0)
        self.assertIn("--questions", terminal.getvalue())

    def test_dataset_library_and_cache(self):
        source = [{"question": "x1", "chosen": "c1", "rejected": "r1"},
                  {"question": "x2", "chosen": "c2", "rejected": "r2"},
                  {"question": "x3", "chosen": "c3", "rejected": "r3"}]
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(train, "load_dataset", return_value=iter(source)) as loader:
            output = Path(temporary)
            rows = train.load_questions(output, 2)
            loader.assert_called_once_with("Intel/orca_dpo_pairs", split="train", streaming=True)
            self.assertEqual([row.id for row in rows], [0, 1])
            self.assertEqual([row.question for row in rows], ["x1", "x2"])
            self.assertEqual(train.load_questions(output, 2), rows)
            self.assertEqual(loader.call_count, 1)
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(train, "load_dataset", return_value=iter(source[:1])):
            with self.assertRaisesRegex(ValueError, "wrong number"):
                train.load_questions(Path(temporary), 2)

    def test_api_interface(self):
        raw = {**completion("answer"), "created": 0, "id": "test",
               "object": "chat.completion", "system_fingerprint": None,
               "provider": "test-provider"}
        raw["choices"][0]["index"] = 0
        raw["choices"][0]["message"]["role"] = "assistant"
        raw["usage"]["cost"] = 0.001
        client_class = httpx.Client
        handler = Mock(return_value=httpx.Response(200, json=raw))
        with patch.dict(os.environ, {"OPENROUTER_KEY": "dummy"}), \
                patch.object(model_api.httpx, "Client", side_effect=lambda **kwargs:
                             client_class(transport=httpx.MockTransport(handler), **kwargs)):
            result = model_api.call_gen("model", [{"role": "user", "content": "x"}])
        self.assertEqual(result, raw)
        self.assertEqual(model_api.response_text(result), "answer")
        self.assertEqual(handler.call_count, 1)
        request = handler.call_args.args[0]
        self.assertEqual(str(request.url), "https://openrouter.ai/api/v1/chat/completions")
        payload = json.loads(request.content)
        self.assertEqual(payload["messages"], [{"role": "user", "content": "x"}])
        self.assertEqual(payload["max_tokens"], 8192)
        self.assertEqual(payload["temperature"], 0.3)
        for status in (429, 503):
            handler = Mock(return_value=httpx.Response(status, json={"error": {
                "code": status, "message": "Unavailable"}}))
            with patch.dict(os.environ, {"OPENROUTER_KEY": "dummy"}), \
                    patch.object(model_api.httpx, "Client", side_effect=lambda **kwargs:
                                 client_class(transport=httpx.MockTransport(handler), **kwargs)):
                with self.assertRaisesRegex(RuntimeError, f"HTTP {status}"):
                    model_api.call_gen("model", [{"role": "user", "content": "x"}])
            self.assertEqual(handler.call_count, 1)

    def test_invalid_responses(self):
        for result, error in (({}, KeyError), ({"choices": []}, IndexError),
                              ({"choices": None}, TypeError)):
            with self.assertRaises(error):
                model_api.response_text(result)
        for result in (completion(""), {**completion("answer"), "usage": {}}):
            with self.assertRaises(ValueError):
                model_api.response_text(result)
        truncated = completion("partial")
        truncated["choices"][0]["finish_reason"] = "length"
        with self.assertRaises(ValueError):
            model_api.response_text(truncated)
        self.assertEqual(model_api.response_text(truncated, allow_truncated=True), "partial")

    def test_sdk_schema_mismatch_keeps_raw_reply(self):
        raw = {**completion("answer"), "created": "bad metadata", "id": "test",
               "object": "chat.completion", "system_fingerprint": None}
        raw["choices"][0]["index"] = 0
        raw["choices"][0]["message"]["role"] = "assistant"
        client_class = httpx.Client
        handler = Mock(return_value=httpx.Response(200, json=raw))
        with tempfile.TemporaryDirectory() as temporary, \
                patch.dict(os.environ, {"OPENROUTER_KEY": "dummy"}), \
                patch.object(model_api.httpx, "Client", side_effect=lambda **kwargs:
                             client_class(transport=httpx.MockTransport(handler), **kwargs)):
            output = Path(temporary)
            (output / "calls").mkdir()
            messages = [{"role": "user", "content": "x"}]
            self.assertEqual(train.recorded_call(output, 0, "y1", "model", messages), "answer")
            self.assertEqual(json.loads((output / "calls/0_y1.json").read_text())["response"], raw)
            self.assertEqual(train.recorded_call(output, 0, "y1", "model", messages), "answer")
            self.assertEqual(handler.call_count, 1)

    def test_invalid_decision_usage_keeps_summary(self):
        raw = {"answers": {"preference": {"type": "choice", "choice": "option1"}},
               "usage": {"input_tokens": 100}}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            train.save_json(output / "dataset.json", {"rows": [
                {"id": 0, "question": "x", "chosen": "c", "rejected": "r"}]})

            def fake_call(model, messages, **settings):
                return raw if settings.get("json_output") else completion(f"y{api.call_count}")

            with patch.object(workflow, "call_gen", side_effect=fake_call) as api, \
                        patch.object(workflow, "call_judge", side_effect=lambda state, rubric, model:
                                     api(model, [{"role": "system", "content": rubric},
                                                 {"role": "user", "content": json.dumps(state)}], json_output=True)), \
                    patch("sys.argv", ["train.py", "--questions", "1", "--output-dir", str(output)]), \
                    redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as failure:
                    train.main()
            self.assertEqual(failure.exception.code, 1)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["losses"]["N"], 1)
            self.assertEqual(summary["losses"]["M"], 0)
            self.assertEqual(len(summary["errors"]), 1)
            self.assertEqual(summary["usage"]["prompt_tokens"], 130)
            self.assertEqual(summary["usage"]["total_tokens"], 45)
            self.assertEqual(json.loads((output / "calls/0_judge_y1_y2.json").read_text())["response"], raw)

    def test_decision_api_interface(self):
        raw = {"model": "microsoft/microsoft-decision-1", "provider": "Azure",
               "answers": {"preference": {"type": "choice", "choice": "option2",
                                          "confidence": 0.6,
                                          "probabilities": {"option1": 0.2, "option2": 0.8}}},
               "usage": {"input_tokens": 100, "output_tokens": 0, "cost": 0.0000042}}
        client_class = httpx.Client
        handler = Mock(return_value=httpx.Response(200, json=raw))
        with patch.dict(os.environ, {"OPENROUTER_KEY": "dummy"}), \
                patch.object(model_api.httpx, "Client", side_effect=lambda **kwargs:
                             client_class(transport=httpx.MockTransport(handler), **kwargs)):
            result = model_api.call_judge({"candidate_1": "a", "candidate_2": "b"}, "Compare quality")
        self.assertEqual(result, raw)
        self.assertEqual(handler.call_count, 1)
        request = handler.call_args.args[0]
        self.assertEqual(str(request.url), "https://openrouter.ai/api/alpha/decisions")
        payload = json.loads(request.content)
        self.assertEqual(set(payload), {"model", "state", "questions"})
        self.assertEqual(payload["state"], {"candidate_1": "a", "candidate_2": "b"})
        self.assertEqual(payload["questions"]["preference"]["instructions"], "Compare quality")
        self.assertEqual(set(payload["questions"]["preference"]["criteria"]), {"option1", "option2"})
        self.assertEqual(json.loads(model_api.response_text(result)), {"choice": "option2"})
        for choice in ("unknown", "tie"):
            raw["answers"]["preference"]["choice"] = choice
            with self.assertRaises(ValueError):
                model_api.response_text(raw)

    def test_ten_questions_and_cache(self):
        def fake_call(model, messages, **settings):
            self.assertEqual([m["role"] for m in messages], ["system", "user"])
            if settings.get("json_output"):
                task = json.loads(messages[1]["content"])
                self.assertIn("instruction_policy", task)
                choice = "option1" if int(task["candidate_1"].split()[-1]) < int(task["candidate_2"].split()[-1]) else "option2"
                return completion(json.dumps({"choice": choice}))
            self.assertEqual(settings["max_tokens"], 8192)
            self.assertEqual(settings["temperature"], 0.3)
            return completion(f"sample {api.call_count}")

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            rows = [{"id": i, "question": f"question {i}", "system": "",
                     "chosen": "chosen", "rejected": "rejected"} for i in range(10)]
            train.save_json(output / "dataset.json", {"rows": rows})
            with patch.object(workflow, "call_gen", side_effect=fake_call) as api, \
                        patch.object(workflow, "call_judge", side_effect=lambda state, rubric, model:
                                     api(model, [{"role": "system", "content": rubric},
                                                 {"role": "user", "content": json.dumps(state)}], json_output=True)), \
                    patch("sys.argv", ["train.py", "--output-dir", str(output)]), \
                    redirect_stdout(io.StringIO()):
                train.main()
                self.assertEqual(api.call_count, 60)
                train.main()  # Completed requests are reused without additional calls.
                self.assertEqual(api.call_count, 60)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["completed_questions"], 10)
            self.assertEqual(summary["usage"]["total_tokens"], 900)
            self.assertIsNone(summary["final_judge"])
            self.assertEqual(summary["losses"]["status"], "likelihood_backend_pending")
            self.assertIsNone(summary["losses"]["L_total"])
            self.assertEqual(summary["update"]["optimizer_steps"], 0)
            preferences = json.loads((output / "intermediate_preferences.json").read_text())
            self.assertEqual(len(preferences), 30)
            for row in preferences:
                self.assertNotEqual(row["y_chosen"], row["y_rejected"])
                self.assertNotIn("confidence", row)
                self.assertEqual(row["a_messages"][1]["content"], row["x"])
                self.assertIn("system", row)
            requests = TypeAdapter(list[ScoringRequest]).validate_json((output / "scoring_requests.json").read_text())
            for row, request in zip(rows, requests):
                record = json.loads((output / f"question_{row['id']}.json").read_text())
                self.assertNotIn("z1", record)
                self.assertNotIn("z2", record)
                self.assertEqual(request.final_answers, [row["chosen"], row["rejected"]])
                for i in range(3):
                    self.assertEqual(json.loads(request.b_messages[i][-1]["content"])["draft"],
                                     record[f"y{i+1}"])
                calls = list((output / "calls").glob(f"{row['id']}_*.json"))
                self.assertEqual(len(calls), 6)
            scores = {"reduction": "sum", "include_eos": True, "rows": [
                {"id": r.id, "request_sha256": r.request_sha256,
                 "a": [-1., -2., -3.], "a_ref": [-1., -2., -3.],
                 "b": [[-1., -2.], [-3., -1.], [-2., -1.]], "b_ref": [[-1., -2.], [-3., -1.], [-2., -1.]]}
                for r in requests]}
            score_path = output / "scores.json"
            train.save_json(score_path, scores)
            typed_preferences = TypeAdapter(list[IntermediatePreference]).validate_python(preferences)
            metrics = compute_saved_losses(score_path, requests, typed_preferences)
            import math
            self.assertAlmostEqual(metrics["L_sys"], math.log(2))
            self.assertAlmostEqual(metrics["L_A_weighted"], 0.5 * math.log(2))
            self.assertEqual(metrics["M"], 30)
            wrong_pair = IntermediatePreference(**{**preferences[0], "y_rejected": "not a scored candidate"})
            with self.assertRaisesRegex(ValueError, "Rejected intermediate"):
                compute_saved_losses(score_path, requests, [wrong_pair])
            with patch.object(workflow, "call_gen", side_effect=AssertionError("Unexpected API call")), \
                    patch.object(workflow, "call_judge", side_effect=AssertionError("Unexpected judge call")), \
                    patch("sys.argv", ["train.py", "--output-dir", str(output), "--scores", str(score_path)]), \
                    redirect_stdout(io.StringIO()) as terminal:
                train.main()
                self.assertIn("L_sys=0.693147", terminal.getvalue())
                self.assertIn("M_intermediate=30", terminal.getvalue())
            scores["rows"][0]["request_sha256"] = "wrong"
            train.save_json(score_path, scores)
            with self.assertRaises(ValueError):
                compute_saved_losses(score_path, requests, typed_preferences)

            # Reusing a request after changing its text must reject the old scores.
            scores["rows"][0]["request_sha256"] = requests[0].request_sha256
            train.save_json(score_path, scores)
            requests[0].final_answers[0] = "different chosen answer"
            with self.assertRaisesRegex(ValueError, "hash does not match"):
                compute_saved_losses(score_path, requests, typed_preferences)

            # Invalid labels stay invalid on replay; no repair or repeated judge call.
            with patch.object(workflow, "recorded_call", return_value='{"choice":"uncertain"}'):
                with self.assertRaises(ValueError):
                    train.judge_pair(output, DatasetRow(**rows[0]), "a", "b", "judge_y", "y1", "y2",
                                     train.load_prompts(train.PROMPT_FILE))

    def test_scoring_failure_saves_current_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            train.save_json(output / "dataset.json", {"rows": [
                {"id": 0, "question": "x", "chosen": "c", "rejected": "r"}]})
            # An older successful result must not survive the failed scoring run.
            train.save_json(output / "summary.json", {"status": "old success"})
            train.save_json(output / "losses.json", {"L_total": 123})
            scores = output / "scores.json"
            train.save_json(scores, {"reduction": "mean", "include_eos": True, "rows": []})

            def fake_call(model, messages, **settings):
                if settings.get("json_output"):
                    return completion('{"choice":"option1"}')
                return completion(f"sample {api.call_count}")

            argv = ["train.py", "--questions", "1", "--output-dir", str(output), "--scores", str(scores)]
            with patch.object(workflow, "call_gen", side_effect=fake_call) as api, \
                        patch.object(workflow, "call_judge", side_effect=lambda state, rubric, model:
                                     api(model, [{"role": "system", "content": rubric},
                                                 {"role": "user", "content": json.dumps(state)}], json_output=True)), \
                    patch("sys.argv", argv), redirect_stdout(io.StringIO()) as terminal:
                with self.assertRaises(SystemExit) as result:
                    train.main()

            self.assertEqual(result.exception.code, 1)
            self.assertEqual(api.call_count, 6)
            summary = json.loads((output / "summary.json").read_text())
            losses = json.loads((output / "losses.json").read_text())
            self.assertEqual(summary["losses"], losses)
            self.assertEqual(losses["status"], "loss_evaluation_failed")
            self.assertIsNone(losses["L_total"])
            self.assertEqual(summary["usage"]["total_tokens"], 90)
            self.assertEqual(summary["errors"][0]["stage"], "scoring")
            self.assertEqual(summary["update"]["optimizer_steps"], 0)
            self.assertIn("Scoring failed:", terminal.getvalue())
            self.assertIn("total=90", terminal.getvalue())

    def test_independent_collection_and_incomplete_outputs(self):
        for failure in ("judge_y", "y1", "y2", "y3"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                train.save_json(output / "dataset.json", {"rows": [{"id": 0, "question": "x",
                    "system": "Explain in detail", "chosen": "correct", "rejected": "wrong"}]})
                def fake_call(model, messages, **settings):
                    if settings.get("json_output"):
                        task = json.loads(messages[1]["content"])
                        self.assertEqual(task["comparison_type"], "intermediate")
                        stage = "judge_y"
                        if stage == failure:
                            raise RuntimeError("simulated judge failure")
                        return completion('{"choice":"option1"}')
                    stage = f"y{api.call_count}"
                    result = completion(stage)
                    if stage == failure:
                        result["choices"][0]["finish_reason"] = "length"
                    return result
                with patch.object(workflow, "call_gen", side_effect=fake_call) as api, \
                        patch.object(workflow, "call_judge", side_effect=lambda state, rubric, model:
                                     api(model, [{"role": "system", "content": rubric},
                                                 {"role": "user", "content": json.dumps(state)}], json_output=True)), \
                        patch("sys.argv", ["train.py", "--questions", "1", "--output-dir", str(output)]), \
                        redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        train.main()
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(summary["update"]["system_rows"], 1)
                self.assertEqual(summary["completed_questions"], 0)
                prefs = json.loads((output / "intermediate_preferences.json").read_text())
                self.assertEqual(len(prefs), 0)
                record = json.loads((output / "question_0.json").read_text())
                if failure in ("y1", "y2", "y3"):
                    self.assertEqual(record["intermediate_judgments"], [])
                    self.assertEqual(record[failure], failure)  # Partial text still recorded.
                if failure == "judge_y":
                    self.assertIn("scoring_request", record)
                else:
                    self.assertNotIn("scoring_request", record)

    def test_prompt_changes_do_not_overwrite_existing_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            prompts = train.load_prompts(train.PROMPT_FILE)
            prompts.a_instruction = "Custom draft instruction"
            prompts.b_instruction = "Custom synthesis instruction"
            custom = output / "custom.json"
            train.save_json(custom, prompts)
            rows = [{"id": 0, "question": "x", "system": "original system",
                     "chosen": "correct", "rejected": "wrong"}]
            train.save_json(output / "dataset.json", {"rows": rows})
            def fake_call(model, messages, **settings):
                if settings.get("json_output"):
                    return completion('{"choice":"tie"}')
                self.assertIn("original system", messages[0]["content"])
                return completion(str(api.call_count))
            argv = ["train.py", "--questions", "1", "--prompt-file", str(custom), "--output-dir", str(output)]
            with patch.object(workflow, "call_gen", side_effect=fake_call) as api, \
                        patch.object(workflow, "call_judge", side_effect=lambda state, rubric, model:
                                     api(model, [{"role": "system", "content": rubric},
                                                 {"role": "user", "content": json.dumps(state)}], json_output=True)), \
                    patch("sys.argv", argv), redirect_stdout(io.StringIO()):
                train.main()
                request = json.loads((output / "scoring_requests.json").read_text())[0]
                for messages in request["b_messages"]:
                    self.assertEqual(messages[0]["role"], "system")
                    self.assertIn("original system", messages[0]["content"])
                    self.assertIn("Custom synthesis instruction", messages[0]["content"])
                    self.assertNotIn("Custom draft instruction", json.dumps(messages))
                    self.assertIn(prompts.instruction_policy, messages[0]["content"])
                original_record = (output / "question_0.json").read_text()
                prompts.b_instruction = "Changed synthesis role"
                train.save_json(custom, prompts)
                with redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                    train.main()
                self.assertEqual(api.call_count, 4)
                self.assertEqual((output / "question_0.json").read_text(), original_record)


if __name__ == "__main__":
    unittest.main()
