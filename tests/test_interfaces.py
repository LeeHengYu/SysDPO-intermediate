"""Shared schema and judge API transport checks without network calls."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import httpx
from pydantic import TypeAdapter, ValidationError

from core import model_api, workflow
from core.schemas import DatasetRow, Judgment, Prompts, QuestionRecord, ScoringRequest


def decision(choice="option1"):
    return {"model": "microsoft/microsoft-decision-1", "provider": "Azure",
            "answers": {"preference": {"type": "choice", "choice": choice}},
            "usage": {"input_tokens": 100, "output_tokens": 0, "cost": 0.0000042}}


class InterfaceChecks(unittest.TestCase):
    def test_preferred_option_maps_back_to_the_candidate_id(self):
        prompts = workflow.load_prompts(workflow.PROMPT_FILE)
        row = DatasetRow(id=7, question="question", chosen="yes", rejected="no")
        for choice, winner_index in (("option1", 0), ("option2", 1)):
            with self.subTest(choice=choice), patch("core.workflow.recorded_call",
                    return_value=json.dumps({"choice": choice})):
                judged = workflow.judge_pair(Path("unused"), row, "one", "two",
                                             "judge_y1_y2", "y1", "y2", prompts)
            self.assertEqual(judged.winner, judged.presented_order[winner_index])

    def test_scoring_request_hash_and_shape_validation(self):
        original = {"id": 0, "a_messages": [], "intermediates": ["y1", "y2"],
                    "b_messages": [[], []], "final_answers": ["c", "r"], "candidate_mask": [True, True],
                    "reference_intermediates": ["ref1", "ref2"], "reference_b_messages": [[], []],
                    "reference_candidate_mask": [True, True]}
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
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            path = Path(temporary) / "question.json"
            record = QuestionRecord(id=0, x="x", system="", z_chosen="c", z_rejected="r", a_messages=[])
            record.intermediate_judgments = [Judgment(choice="tie")]
            workflow.save_json(path, record)
            saved = json.loads(path.read_text())
            self.assertNotIn("y1", saved)
            self.assertNotIn("scoring_request", saved)
            self.assertIsNone(saved["intermediate_judgments"][0]["winner"])

    def test_api_interface(self):
        raw = decision()
        client_class = httpx.Client
        handler = Mock(return_value=httpx.Response(200, json=raw))
        with patch.dict(os.environ, {"OPENROUTER_KEY": "dummy"}), \
                patch.object(model_api.httpx, "Client", side_effect=lambda **kwargs:
                             client_class(transport=httpx.MockTransport(handler), **kwargs)):
            result = model_api.call_judge({"candidate_1": "a", "candidate_2": "b"}, "Compare quality")
        self.assertEqual(result, raw)
        self.assertEqual(json.loads(model_api.response_text(result)), {"choice": "option1"})
        self.assertEqual(handler.call_count, 1)
        request = handler.call_args.args[0]
        self.assertEqual(str(request.url), "https://openrouter.ai/api/alpha/decisions")
        payload = json.loads(request.content)
        self.assertEqual(payload["model"], "microsoft/microsoft-decision-1")

        for status in (401, 503):
            handler = Mock(return_value=httpx.Response(status, json={"error": {
                "code": status, "message": "Unavailable"}}))
            with patch.dict(os.environ, {"OPENROUTER_KEY": "dummy"}), \
                    patch.object(model_api.httpx, "Client", side_effect=lambda **kwargs:
                                 client_class(transport=httpx.MockTransport(handler), **kwargs)):
                with self.assertRaisesRegex(RuntimeError, f"HTTP {status}"):
                    model_api.call_judge({"candidate_1": "a", "candidate_2": "b"}, "Compare quality")
            self.assertEqual(handler.call_count, 1)

    def test_rate_limit_waits_30_seconds_before_each_retry(self):
        raw = decision()
        client_class = httpx.Client
        handler = Mock(side_effect=[
            httpx.Response(429, json={"error": {"code": 429, "message": "Rate limited"}}),
            httpx.Response(429, json={"error": {"code": 429, "message": "Rate limited"}}),
            httpx.Response(200, json=raw),
        ])
        with patch.dict(os.environ, {"OPENROUTER_KEY": "dummy"}), \
                patch.object(model_api.httpx, "Client", side_effect=lambda **kwargs:
                             client_class(transport=httpx.MockTransport(handler), **kwargs)), \
                patch.object(model_api.time, "sleep") as sleep:
            result = model_api.call_judge({"candidate_1": "a", "candidate_2": "b"}, "Compare quality")

        self.assertEqual(result, raw)
        self.assertEqual(handler.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertTrue(all(call.args == (30,) for call in sleep.call_args_list))

    def test_invalid_responses(self):
        for result, error in (({}, KeyError), ({"answers": {}}, KeyError),
                              ({"answers": None}, TypeError)):
            with self.assertRaises(error):
                model_api.response_text(result)
        for result in (decision("tie"), {**decision(), "usage": {}}, {"error": "Unavailable"}):
            with self.assertRaises(ValueError):
                model_api.response_text(result)

    def test_judge_cache_keeps_raw_reply_and_request_hash(self):
        raw = decision()
        client_class = httpx.Client
        handler = Mock(return_value=httpx.Response(200, json=raw))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary, \
                patch.dict(os.environ, {"OPENROUTER_KEY": "dummy"}), \
                patch.object(model_api.httpx, "Client", side_effect=lambda **kwargs:
                             client_class(transport=httpx.MockTransport(handler), **kwargs)):
            output = Path(temporary)
            (output / "calls").mkdir()
            messages = [{"role": "system", "content": "Compare quality"},
                        {"role": "user", "content": json.dumps({"candidate_1": "a", "candidate_2": "b"})}]
            settings = {"temperature": 0, "max_tokens": 512, "json_output": True}
            request = {"model": workflow.JUDGE, "messages": messages, **settings}
            expected_hash = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
            for _ in range(2):
                result = workflow.recorded_call(output, 0, "judge_y1_y2", workflow.JUDGE, messages, **settings)
                self.assertEqual(json.loads(result), {"choice": "option1"})

            saved = json.loads((output / "calls/0_judge_y1_y2.json").read_text())
            self.assertEqual(saved["response"], raw)
            self.assertEqual(saved["request_sha256"], expected_hash)
            self.assertEqual(handler.call_count, 1)

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


if __name__ == "__main__":
    unittest.main()
