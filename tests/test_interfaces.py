"""Shared schema and API transport checks; no smoke runner or network calls."""

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
from core.schemas import Judgment, Prompts, QuestionRecord, ScoringRequest


def completion(text):
    return {"model": "test-model", "choices": [{"message": {"content": text},
                                                 "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


class InterfaceChecks(unittest.TestCase):
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
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary, \
                patch.dict(os.environ, {"OPENROUTER_KEY": "dummy"}), \
                patch.object(model_api.httpx, "Client", side_effect=lambda **kwargs:
                             client_class(transport=httpx.MockTransport(handler), **kwargs)):
            output = Path(temporary)
            (output / "calls").mkdir()
            messages = [{"role": "user", "content": "x"}]
            self.assertEqual(workflow.recorded_call(output, 0, "y1", "model", messages), "answer")
            self.assertEqual(json.loads((output / "calls/0_y1.json").read_text())["response"], raw)
            self.assertEqual(workflow.recorded_call(output, 0, "y1", "model", messages), "answer")
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
