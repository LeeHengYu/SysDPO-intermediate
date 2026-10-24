"""Real tiny Qwen/LoRA tests, requiring no model downloads or API calls."""
from copy import deepcopy
from pathlib import Path
import json
import random
import shutil
import unittest
import uuid
from unittest.mock import patch

import torch
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model

from core.training import TrainingConfig, backward_row, copy_checkpoint, restore_rng, save_checkpoint, train_collection
from core.likelihood import preference_losses, score_request
from core.schemas import ScoringRequest, IntermediatePreference
from core.workflow import save_json
from multi_llm.runs import prepare_training


class TinyTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        return [1, 3, 4]

    def encode(self, text, **kwargs):
        return [5 + ord(char) % 20 for char in text]


class OfflineTests(unittest.TestCase):
    def test_cuda_rng_follows_worker_when_node_layout_changes(self):
        first, second = torch.tensor([1], dtype=torch.uint8), torch.tensor([2], dtype=torch.uint8)
        rng = {"torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
               "cuda_rng": [first, second]}
        for saved, expected in ((rng, second),
                                ({**rng, "cuda_rng": [first], "cuda_device": 0}, first)):
            with self.subTest(saved_devices=len(saved["cuda_rng"])), \
                    patch("core.training.torch.cuda.set_rng_state") as restore:
                restore_rng({"rank_rng_states": [rng, saved]}, rank=1, device="cuda")
                self.assertIs(restore.call_args.args[0], expected)

    def test_real_optimizer_resume_matches_uninterrupted(self):
        root = Path(__file__).parent / ("resume_test_" + uuid.uuid4().hex)
        base, collection = root / "base", root / "collection"
        base.mkdir(parents=True)
        collection.mkdir()
        try:
            torch.manual_seed(7)
            backend = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "question": 2,
                "one": 3, "two": 4, "three": 5, "yes": 6, "no": 7}, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend,
                unk_token="[UNK]", eos_token="[EOS]", pad_token="[EOS]")
            tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
            tokenizer.save_pretrained(base)
            Qwen2ForCausalLM(Qwen2Config(vocab_size=8, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)).save_pretrained(base)
            save_json(base / "provenance.json", {"revision": "tiny-test"})
            original = {k: v.clone() for k, v in load_file(str(base / "model.safetensors")).items()}
            messages = [{"role": "user", "content": "question"}]
            requests = [ScoringRequest(id=i, a_messages=messages, intermediates=["one", "two", "three"],
                b_messages=[messages] * 3, final_answers=["yes", "no"], candidate_mask=[True] * 3) for i in range(4)]
            pairs = [IntermediatePreference(id=i, x="question", system="", a_messages=messages,
                y_chosen="one", y_rejected="two", finish_reasons={}) for i in range(4)]
            save_json(collection / "scoring_requests.json", requests)
            save_json(collection / "intermediate_preferences.json", pairs)
            config = TrainingConfig(max_steps=3, optimizer_rows=1, checkpoint_rows=2, learning_rate=1e-3,
                                    warmup_steps=0, rank=2, alpha=2, dropout=0.1)
            full = train_collection(base, collection, root / "work", root / "full", config, device="cpu")
            resumed = train_collection(base, collection, root / "work_resume", root / "resumed", config,
                                       resume=root / "full" / "rows_000002", device="cpu")
            state = json.loads((full / "state.json").read_text())
            self.assertEqual(state["optimizer_steps"], 3)
            self.assertEqual(state["rows_done"], 3)
            self.assertEqual(state["total_rows"], 4)
            self.assertEqual(state["stop_reason"], "max_steps")
            audits = [json.loads(path.read_text()) for path in sorted((root / "full/metrics").glob("*.json"))]
            self.assertEqual([row["batch_rows"] for row in audits], [1, 1, 1])
            for row in audits:
                self.assertAlmostEqual(row["L_total"], row["L_sys"] + 0.5 * row["L_A"])
            self.assertEqual(state["last_audit"], audits[-1])
            with patch("core.training.load_models") as load:
                self.assertEqual(train_collection(base, collection, root / "extra_work", root / "extra", config,
                                                 resume=full, device="cpu"), full)
                load.assert_not_called()
            for role in ("A", "B"):
                weights = {k: v.clone() for k, v in load_file(str(full / f"{role}_adapter" / "adapter_model.safetensors")).items()}
                restored = {k: v.clone() for k, v in load_file(str(resumed / f"{role}_adapter" / "adapter_model.safetensors")).items()}
                for name in weights:
                    torch.testing.assert_close(weights[name], restored[name], atol=0, rtol=0)
                self.assertTrue(any(torch.count_nonzero(value) for name, value in weights.items() if "lora_B" in name))
            for name, value in {k: v.clone() for k, v in load_file(str(base / "model.safetensors")).items()}.items():
                torch.testing.assert_close(value, original[name], atol=0, rtol=0)

            carry_config = TrainingConfig(max_steps=1, optimizer_rows=3, checkpoint_rows=3,
                                          warmup_steps=0, rank=2, alpha=2)
            first, second = root / "carry/round_000", root / "carry/round_001"
            for stage, subset in ((first, slice(0, 2)), (second, slice(2, 4))):
                stage.mkdir(parents=True)
                save_json(stage / "scoring_requests.json", requests[subset])
                save_json(stage / "intermediate_preferences.json", pairs[subset])
            prepare_training(first)
            pending = train_collection(base, first / "training", root / "work_pending",
                                       first / "checkpoints", carry_config, device="cpu")
            state = json.loads((pending / "state.json").read_text())
            self.assertEqual(state["optimizer_steps"], 0)
            self.assertEqual(state["rows_done"], 0)
            self.assertEqual(state["stop_reason"], "collection_remainder")

            prepare_training(second, pending)
            prepare_training(second, pending)
            merged = json.loads((second / "training/scoring_requests.json").read_text())
            self.assertEqual([row["id"] for row in merged], [0, 1, 2, 3])
            merged_pairs = json.loads((second / "training/intermediate_preferences.json").read_text())
            self.assertEqual([row["id"] for row in merged_pairs], [0, 1, 2, 3])
            final = train_collection(base, second / "training", root / "work_carry",
                                     second / "checkpoints", carry_config, previous=pending, device="cpu")
            state = json.loads((final / "state.json").read_text())
            self.assertEqual(state["optimizer_steps"], 1)
            self.assertEqual(state["rows_done"], 3)
            self.assertEqual(state["total_rows"], 4)
            self.assertEqual(state["last_audit"]["batch_rows"], 3)
        finally:
            shutil.rmtree(root)

    def test_replayed_gradient_matches_direct_with_dropout(self):
        self.assert_replayed_gradients("cpu", torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA allocation required")
    def test_replayed_gradient_matches_direct_on_cuda(self):
        self.assert_replayed_gradients("cuda", torch.bfloat16)

    def assert_replayed_gradients(self, device, dtype):
        torch.manual_seed(4)
        base = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16,
            intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, attention_dropout=0.0)).to(device=device, dtype=dtype)
        reference = deepcopy(base).eval().requires_grad_(False)
        a = get_peft_model(deepcopy(base), LoraConfig(r=2, lora_alpha=2,
            lora_dropout=0.1, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        b = deepcopy(a)
        a.train()
        b.train()
        for model in (a, b):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        replay_a, replay_b = deepcopy(a), deepcopy(b)
        tokenizer = TinyTokenizer()
        messages = [{"role": "user", "content": "question"}]
        request = ScoringRequest(id=0, a_messages=messages, intermediates=["ab", "cd", "ef"],
            b_messages=[messages] * 3, final_answers=["yes", "no"], candidate_mask=[True] * 3)
        pairs = [(0, 0, 1), (0, 0, 2), (0, 1, 2)]
        rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if device == "cuda" else None
        scores = score_request(request, a, b, reference, reference, tokenizer, tokenizer)
        direct = preference_losses({key: value.unsqueeze(0) for key, value in scores.items()}, pairs)
        loss = direct["L_sys"] / 2 + direct["L_A_weighted"] * 3 / 5
        loss.backward()
        rng_after = torch.get_rng_state()
        cuda_rng_after = torch.cuda.get_rng_state_all() if device == "cuda" else None
        torch.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        replay_loss = backward_row(request, pairs, (replay_a, replay_b, reference), tokenizer,
                                   2, 5, TrainingConfig())
        self.assertAlmostEqual(replay_loss.L_total, loss.item(), places=6)
        self.assertAlmostEqual(replay_loss.L_sys, direct["L_sys"].item() / 2, places=6)
        self.assertAlmostEqual(replay_loss.L_A, direct["L_A"].item() * 3 / 5, places=6)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_after))
        if cuda_rng_after is not None:
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(torch.cuda.get_rng_state_all(), cuda_rng_after)))
        for direct_model, replay_model in ((a, replay_a), (b, replay_b)):
            for p, q in zip(direct_model.parameters(), replay_model.parameters()):
                if p.requires_grad:
                    torch.testing.assert_close(p.grad, q.grad, atol=2e-7, rtol=2e-4)
        self.assertTrue(all(p.grad is None for p in reference.parameters()))

    def test_checkpoint_publishes_only_after_copy_and_restores_state(self):
        root = Path(__file__).parent / ("checkpoint_test_" + uuid.uuid4().hex)
        root.mkdir()
        try:
            config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
            a = get_peft_model(Qwen2ForCausalLM(config), LoraConfig(
                r=2, target_modules=["q_proj"], task_type="CAUSAL_LM"))
            b = deepcopy(a)
            params = [p for model in (a, b) for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.RMSprop(params, lr=1e-3)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1)
            sum(p.sum() for p in params).backward()
            optimizer.step()
            scheduler.step()
            snapshot = save_checkpoint((a, b), optimizer, scheduler, TrainingConfig(),
                                       {"rows_done": 256}, root / "scratch" / "rows_000256")
            self.assertFalse((snapshot / "complete").exists())
            durable = copy_checkpoint(snapshot, root / "durable")
            self.assertTrue((durable / "complete").exists())
            saved = torch.load(durable / "training_state.pt", weights_only=True)
            self.assertEqual(saved["scheduler"], scheduler.state_dict())
            self.assertTrue(torch.equal(saved["torch_rng"], torch.get_rng_state()))
            self.assertEqual(json.loads((durable / "state.json").read_text())["rows_done"], 256)
            with self.assertRaises(FileExistsError):
                copy_checkpoint(snapshot, root / "durable")
        finally:
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
