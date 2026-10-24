"""Held-out evaluation checks without downloads or paid API calls."""

from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from core.workflow import PROMPT_FILE, save_json
from multi_llm.evaluate import (FINAL_CRITERIA, final_comparison, generate,
                                judge, prepare_evaluation)
from tests.test_interfaces import decision


class EvaluationTests(unittest.TestCase):
    def test_split_generation_baseline_and_judgment_reuse(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            (root / "data").mkdir()
            base = root / "base"
            base.mkdir()
            save_json(base / "provenance.json", {"revision": "tiny"})
            current = root / "runs/source/round_000"
            checkpoint = current / "checkpoints/rows_000001"
            checkpoint.mkdir(parents=True)
            (checkpoint / "complete").touch()
            (current / "training.ready").touch()
            save_json(current / "collection_config.json", {
                "base": {"revision": "tiny"}, "prompts": json.loads(PROMPT_FILE.read_text())})
            rows = [{"id": i, "system": "Dataset instruction", "question": f"Question {i}",
                     "chosen": "Chosen answer", "rejected": "Rejected answer"} for i in (1, 2)]
            save_json(root / "data/train.json", rows[:1])
            save_json(root / "data/test.json", rows[1:] + [{**rows[0], "id": 3}])
            output = root / "eval"
            self.assertEqual(prepare_evaluation(root, "source", output, 256), checkpoint)
            manifest = json.loads((output / "evaluation.json").read_text())
            self.assertEqual(manifest["row_ids"], [2])
            self.assertEqual(manifest["training_prompt_overlap_ids"], [3])
            with self.assertRaisesRegex(ValueError, "changed"):
                prepare_evaluation(root, "source", output, 128)
            save_json(root / "data/test.json", rows)
            with self.assertRaisesRegex(ValueError, "disjoint"):
                prepare_evaluation(root, "source", output, 256)

            class Model:
                disabled = False

                def to(self, *args): return self
                def eval(self): return self
                def requires_grad_(self, *args): return self

                @contextmanager
                def disable_adapter(self):
                    self.disabled = True
                    try:
                        yield
                    finally:
                        self.disabled = False

            class Tokenizer:
                eos_token_id = 2

                def apply_chat_template(self, messages, **kwargs):
                    return torch.tensor([[1]])

                def decode(self, tokens, **kwargs):
                    return "trained answer" if tokens[0] == 3 else "base answer"

            seen = []

            def predict(model, tokenizer, messages, max_tokens, device):
                seen.append((model.disabled, messages))
                return {"text": "base answer" if model.disabled else "trained answer",
                        "prompt_tokens": 10, "completion_tokens": 2, "finish_reason": "stop"}

            with patch("transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()), \
                    patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=lambda *a, **k: Model()), \
                    patch("peft.PeftModel.from_pretrained", side_effect=lambda model, path: model), \
                    patch("multi_llm.evaluate.generate_answer", side_effect=predict):
                generate(base, checkpoint, output, device="cpu")
                generate(base, checkpoint, output, device="cpu")
            self.assertEqual([disabled for disabled, _ in seen], [False, False, True, True])
            self.assertIn("trained answer", seen[1][1][1]["content"])
            self.assertIn("base answer", seen[3][1][1]["content"])
            self.assertNotIn("Chosen answer", str(seen))
            self.assertTrue((output / "generation.ready").exists())
            self.assertEqual(len(json.loads((output / "trained.json").read_text())), 1)

            with patch.dict("os.environ", {"OPENROUTER_KEY": "dummy"}), \
                    patch("core.model_api.call_judge", return_value=decision("tie")) as api:
                judge(output, workers=2)
                judge(output, workers=2)
            self.assertEqual(api.call_count, 3)
            self.assertEqual(api.call_args.kwargs["criteria"], FINAL_CRITERIA)
            summary = json.loads((output / "summary.json").read_text())
            self.assertTrue(all(value["win_rate"] == 50 for value in summary["comparisons"].values()))

    def test_order_mapping_and_identical_answer_tie(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            directory = Path(temporary)
            (directory / "calls").mkdir()
            record = {"id": 9, "question": "x", "system": "", "chosen": "c",
                      "trained": {"z": {"text": "t"}}, "prompted": {"z": {"text": "b"}}}
            with patch("core.model_api.call_judge", return_value=decision("option1")):
                score = final_comparison(directory, record, "trained_vs_chosen")
            saved = json.loads((directory / "calls/9_trained_vs_chosen.json").read_text())
            self.assertEqual(score, float(saved["presented_order"][0] == "trained"))
            record["prompted"]["z"]["text"] = "t"
            with patch("core.model_api.call_judge") as api:
                self.assertEqual(final_comparison(directory, record, "trained_vs_prompted"), 0.5)
                api.assert_not_called()

    def test_greedy_generation_uses_completion_only(self):
        from multi_llm.evaluate import generate_answer
        from unittest.mock import Mock

        tokenizer = Mock(eos_token_id=2)
        tokenizer.apply_chat_template.return_value = torch.tensor([[7, 8]])
        tokenizer.decode.return_value = "answer"
        model = Mock()
        model.generate.return_value = torch.tensor([[7, 8, 9, 2]])
        result = generate_answer(model, tokenizer, [], 256, "cpu")
        self.assertEqual(result["completion_tokens"], 2)
        self.assertEqual(result["finish_reason"], "stop")
        self.assertFalse(model.generate.call_args.kwargs["do_sample"])
        tokenizer.decode.assert_called_once_with([9, 2], skip_special_tokens=True)
