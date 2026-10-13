"""Numerical fixtures only: no downloads, API requests or production weights."""

from copy import deepcopy
import math
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from likelihood import LAMBDA_A, completion_logp, dpo_pair_loss, get_probs, preference_losses, score_request
from train import accumulate_gradients, freeze
from schemas import ScoringRequest


QUESTIONS, MESSAGES, ANSWERS = 8, 4, 3

class Producer(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Embedding(QUESTIONS, MESSAGES)

    def forward(self, x):
        return self.logits(x).log_softmax(-1)


class Receiver(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Embedding(QUESTIONS * MESSAGES, ANSWERS)

    def forward(self, x):
        y = torch.arange(MESSAGES)
        return self.logits(x[:, None] * MESSAGES + y).log_softmax(-1)


@torch.no_grad()
def collect_intermediate_pairs(a_snapshot, attempts=128):
    """Two independent ordinary samples per question; mocked offline judge."""
    x = torch.arange(attempts) % QUESTIONS
    draws = torch.multinomial(a_snapshot(x).exp(), 2, replacement=True)
    # shortcut: synthetic judge ranks message IDs; replace with cached API labels
    # for real experiments. These scores say nothing about natural-language quality.
    utility = -(draws - (x % MESSAGES)[:, None]).abs()
    keep = utility[:, 0] != utility[:, 1]  # Skip duplicates and ties; no forced resampling.
    x, draws, utility = x[keep], draws[keep], utility[keep]
    first_wins = utility[:, 0] > utility[:, 1]
    chosen = torch.where(first_wins, draws[:, 0], draws[:, 1])
    rejected = torch.where(first_wins, draws[:, 1], draws[:, 0])
    if not len(x):
        raise ValueError("No intermediate preferences accepted.")
    return x, chosen, rejected


def dpo_loss(logp, ref_logp, chosen, rejected, beta):
    """SysDPO's chosen/rejected log-ratio margin, using stable log inputs."""
    rows = torch.arange(len(chosen))
    # Adapted from SysDPO sysdpo/multi_llm/train.py:985-1010 (Apache-2.0).
    # Inputs are already log probabilities; no exp/log round trip or clamping.
    return dpo_pair_loss(logp[rows, chosen], logp[rows, rejected],
                         ref_logp[rows, chosen], ref_logp[rows, rejected], beta)


def get_system_log_probs(a, b, x):
    # Exact marginal over ALL four messages, in log space. No gradient through
    # discrete sampling. This is a toy oracle, not released SysDPO's approximation.
    return torch.logsumexp(a(x)[:, :, None] + b(x), dim=1)


def compute_system_loss(a, b, a_ref, b_ref, batch, beta=0.5):
    x, chosen, rejected = batch
    with torch.no_grad():
        ref = get_system_log_probs(a_ref, b_ref, x)
    return dpo_loss(get_system_log_probs(a, b, x), ref, chosen, rejected, beta)


def compute_intermediate_loss(a, a_ref, batch, beta=0.5):
    x, chosen, rejected = batch
    with torch.no_grad():
        ref = a_ref(x)
    return LAMBDA_A * dpo_loss(a(x), ref, chosen, rejected, beta)


def get_gradients(a, b):
    return [torch.zeros_like(p) if p.grad is None else p.grad.clone()
            for model in (a, b) for p in model.parameters()]


def zero_grad(a, b):
    a.zero_grad(set_to_none=True)
    b.zero_grad(set_to_none=True)


def run_numerical_smoke():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    a, b = Producer(), Receiver()
    a_ref, b_ref = freeze(a), freeze(b)
    references_before = [p.clone() for model in (a_ref, b_ref) for p in model.parameters()]
    initial_parameters = [p.detach().clone() for model in (a, b) for p in model.parameters()]

    # Synthetic stand-in for existing (x, z_chosen, z_rejected) dataset rows.
    # Final preference labels are fixed; no final judge is called during training.
    x = torch.arange(128) % QUESTIONS
    system_batch = (x, x % ANSWERS, (x + 1) % ANSWERS)
    pool = collect_intermediate_pairs(a_ref)
    assert len(pool[0]) >= 32
    intermediate_batch = tuple(column[:32] for column in pool)
    sys_fn = lambda batch: compute_system_loss(a, b, a_ref, b_ref, batch)
    aux_fn = lambda batch: compute_intermediate_loss(a, a_ref, batch)

    # Identical initial policy/reference => each unweighted DPO loss is log(2).
    torch.testing.assert_close(sys_fn(system_batch), torch.full((128,), math.log(2)))
    torch.testing.assert_close(aux_fn(intermediate_batch).mean(),
                               torch.tensor(LAMBDA_A * math.log(2)))

    # L_A reaches A only; L_sys reaches both; references remain frozen.
    aux_fn(intermediate_batch).mean().backward()
    assert a.logits.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in b.parameters())
    zero_grad(a, b)
    sys_fn(system_batch).mean().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0
               for model in (a, b) for p in model.parameters())

    # Compare accumulation with a single full objective, including uneven chunks.
    zero_grad(a, b)
    (sys_fn(system_batch).mean() + aux_fn(intermediate_batch).mean()).backward()
    expected = get_gradients(a, b)
    for microbatch in (1, 2, 7):
        zero_grad(a, b)
        accumulate_gradients(sys_fn, system_batch, microbatch)
        accumulate_gradients(aux_fn, intermediate_batch, microbatch)
        for actual, target in zip(get_gradients(a, b), expected):
            torch.testing.assert_close(actual, target, rtol=2e-5, atol=1e-7)

    with torch.no_grad():
        before = (sys_fn(system_batch).mean().item(), aux_fn(intermediate_batch).mean().item())
    optimizer = torch.optim.Adam([*a.parameters(), *b.parameters()], lr=0.03)
    for global_step in range(30):
        a.train()
        b.train()
        optimizer.zero_grad(set_to_none=True)
        # Independent N=128 and M=32 denominators, ONE shared optimizer step.
        main_loss = accumulate_gradients(sys_fn, system_batch, microbatch=2)
        intermediate_loss = accumulate_gradients(aux_fn, intermediate_batch, microbatch=2)
        total_loss = main_loss + intermediate_loss  # Detached logging values.
        assert math.isfinite(total_loss)
        if not all(torch.isfinite(g).all() for g in get_gradients(a, b)):
            raise FloatingPointError("Non-finite training gradients")
        optimizer.step()
        if global_step in (0, 29):
            print(f"step={global_step + 1:02d} L_sys={main_loss:.6f} "
                  f"weighted_L_A={intermediate_loss:.6f}")
    with torch.no_grad():
        after = (sys_fn(system_batch).mean().item(), aux_fn(intermediate_batch).mean().item())
    assert all(end < start for start, end in zip(before, after)), (before, after)
    for previous, current in zip(initial_parameters, [*a.parameters(), *b.parameters()]):
        assert not torch.equal(previous, current)
    for previous, current in zip(references_before, [*a_ref.parameters(), *b_ref.parameters()]):
        assert torch.equal(previous, current) and current.grad is None
    print(f"Accepted {len(pool[0])}/128 synthetic pairs; N=128, M=32, microbatch=2.")
    print("PASS: accumulation, loss scaling, gradient routing, frozen refs, training.")
    print("CPU toy only: no real LLM/API, tokenizer, LoRA, FP16 or GPU-fit validation.")


class TokenizerFixture:
    eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3]

    def encode(self, text, **kwargs):
        return [4, 5] if text in ("y1", "chosen") else [6]


class ModelFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(8, 7))

    def forward(self, input_ids, attention_mask):
        return {"logits": self.logits[:input_ids.shape[1]][None]}


class ContextTokenizerFixture(TokenizerFixture):
    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 4 if messages[-1]["content"] == "y1" else 6]


class ContextModelFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(2 * torch.eye(7))

    def forward(self, input_ids, attention_mask):
        return {"logits": self.logits[input_ids]}


class LikelihoodChecks(unittest.TestCase):
    def test_masked_nonfinite_logits_are_rejected(self):
        seq = torch.tensor([[1, 2, 3, 4]])
        attn = torch.tensor([[0, 0, 1, 0]])

        # Prompt, padding and unused final positions all need finite logits.
        for position in (0, 2, 3):
            for value in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(position=position, value=value):
                    logits = torch.zeros(1, 4, 7)
                    logits[0, position, 0] = value
                    logits.requires_grad_()

                    with self.assertRaisesRegex(FloatingPointError, "model logits"):
                        get_probs(logits, seq, attn)
                    self.assertIsNone(logits.grad)

    def test_b_scores_use_both_intermediate_contexts(self):
        a, b = ContextModelFixture(), ContextModelFixture()
        tokenizer = ContextTokenizerFixture()
        a_ref, b_ref = freeze(a), freeze(b)
        request = ScoringRequest(
            id=0, a_messages=[{"role": "user", "content": "x"}],
            intermediates=["y1", "y2"],
            b_messages=[[{"role": "user", "content": "y1"}],
                        [{"role": "user", "content": "y2"}]],
            final_answers=["chosen", "rejected"], candidate_mask=[True, True])

        scores = score_request(request, a, b, a_ref, b_ref, tokenizer, tokenizer)

        # This bigram model favors repeating the context token: y1 -> chosen,
        # y2 -> rejected. Reusing either context twice must fail these checks.
        for key in ("b", "b_ref"):
            self.assertGreater(scores[key][0, 0].item(), scores[key][1, 0].item())
            self.assertLess(scores[key][0, 1].item(), scores[key][1, 1].item())
            self.assertGreater(scores[key][0, 0].item(), scores[key][0, 1].item())
            self.assertLess(scores[key][1, 0].item(), scores[key][1, 1].item())

    def test_get_probs_matches_source_values_and_gradients(self):
        torch.manual_seed(11)
        logits = torch.randn(2, 7, 9, requires_grad=True)
        source_logits = logits.detach().clone().requires_grad_()
        seq = torch.tensor([[1, 2, 3, 4, 5, 6, 0], [1, 2, 3, 4, 0, 0, 0]])
        attn = torch.tensor([[0, 0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0, 0]])
        # Released get_probs, finite inputs above eps=1e-8 (train.py:663-679).
        probs = source_logits[:, :-1, :].softmax(-1)
        probs = probs.gather(-1, seq[:, 1:, None]).squeeze(-1)
        mask = attn[:, 1:]
        expected = ((probs * mask).clamp(min=1e-8).log() * mask).sum(-1) / mask.sum(-1)
        actual_probs, actual = get_probs(logits, seq, attn)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_probs, expected.exp())
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(logits.grad, source_logits.grad)
        self.assertEqual(logits.grad[0, :2].count_nonzero(), 0)
        self.assertEqual(logits.grad[1, 3:].count_nonzero(), 0)
        with self.assertRaises(ValueError):
            get_probs(logits, seq, torch.zeros_like(attn))
        bad_logits = logits.detach().clone()
        bad_logits[0, 2, :] = float("nan")
        with self.assertRaises(FloatingPointError):
            get_probs(bad_logits, seq, attn)

    def test_system_matches_source_product_sum_and_gradients(self):
        a = torch.tensor([[-1., -2.], [-1.5, -0.5]], requires_grad=True)
        b = torch.tensor([[[-1., -2.], [-3., -1.]], [[-2., -1.], [-1., -3.]]], requires_grad=True)
        a_ref, b_ref = a.detach() - 0.2, b.detach() - 0.3
        source_a = a.detach().clone().requires_grad_()
        source_b = b.detach().clone().requires_grad_()
        # Released candidate loop (train.py:939-1010), using shared test support.
        policy = sum(source_a[:, k, None].exp() * source_b[:, k].exp() for k in range(2)).log()
        reference = sum(a_ref[:, k, None].exp() * b_ref[:, k].exp() for k in range(2)).log()
        expected = dpo_pair_loss(policy[:, 0], policy[:, 1], reference[:, 0], reference[:, 1]).mean()
        actual = preference_losses({"a": a, "b": b, "a_ref": a_ref, "b_ref": b_ref}, [])["L_sys"]
        torch.testing.assert_close(actual, expected)
        actual.backward()
        expected.backward()
        torch.testing.assert_close(a.grad, source_a.grad)
        torch.testing.assert_close(b.grad, source_b.grad)

    def test_numerical_training_and_accumulation(self):
        run_numerical_smoke()

    def test_completion_mask_and_eos(self):
        model, tokenizer = ModelFixture(), TokenizerFixture()
        value = completion_logp(model, tokenizer, [{"role": "user", "content": "x"}], "chosen", reduction="sum")
        self.assertAlmostEqual(value.item(), -3 * math.log(7), places=5)
        value.backward()
        self.assertEqual(model.logits.grad[:2].count_nonzero(), 0)
        self.assertGreater(model.logits.grad[2:5].abs().sum(), 0)
        self.assertEqual(model.logits.grad[5:].count_nonzero(), 0)
        mean = completion_logp(model, tokenizer, [], "chosen")
        self.assertAlmostEqual(mean.item(), -math.log(7), places=5)

    def test_scoring_and_frozen_reference(self):
        a, b, tokenizer = ModelFixture(), ModelFixture(), TokenizerFixture()
        a_ref, b_ref = deepcopy(a).requires_grad_(False), deepcopy(b).requires_grad_(False)
        request = ScoringRequest(id=0, a_messages=[], intermediates=["y1", "y2"],
                                 b_messages=[[], []], final_answers=["chosen", "rejected"],
                                 candidate_mask=[True, True])
        scores = score_request(request, a, b, a_ref, b_ref, tokenizer, tokenizer)
        scores = {k: value[None] for k, value in scores.items()}
        losses = preference_losses(scores, [(0, 0, 1)])
        self.assertAlmostEqual(losses["L_sys"].item(), math.log(2), places=6)
        self.assertAlmostEqual(losses["L_A_weighted"].item(), 0.5 * math.log(2), places=6)
        losses["L_total"].backward()
        self.assertIsNotNone(a.logits.grad)
        self.assertIsNotNone(b.logits.grad)
        self.assertIsNone(a_ref.logits.grad)
        self.assertIsNone(b_ref.logits.grad)

        request.b_messages[0].append({"role": "user", "content": "changed context"})
        with self.assertRaisesRegex(ValueError, "hash does not match"):
            score_request(request, a, b, a_ref, b_ref, tokenizer, tokenizer)

    def test_pair_average_and_gradient_routing(self):
        a = torch.tensor([[-1., -2.], [-2., -1.]], requires_grad=True)
        b = torch.tensor([[[-1., -2.], [-3., -1.]], [[-2., -1.], [-1., -3.]]], requires_grad=True)
        scores = {"a": a, "b": b, "a_ref": a.detach().clone().requires_grad_(),
                  "b_ref": b.detach().clone().requires_grad_()}
        losses = preference_losses(scores, [(0, 0, 1), (1, 1, 0)])
        self.assertAlmostEqual(losses["L_A_weighted"].item(), 0.5 * math.log(2), places=6)
        losses["L_A_weighted"].backward(retain_graph=True)
        self.assertGreater(a.grad.abs().sum(), 0)
        self.assertIsNone(b.grad)
        self.assertIsNone(scores["a_ref"].grad)
        self.assertIsNone(scores["b_ref"].grad)
        a.grad = None
        losses["L_sys"].backward()
        self.assertGreater(a.grad.abs().sum(), 0)
        self.assertGreater(b.grad.abs().sum(), 0)
        empty = preference_losses(scores, [])
        self.assertEqual(empty["M"], 0)
        self.assertEqual(empty["L_A_weighted"].item(), 0)
        with self.assertRaises(ValueError):
            preference_losses(scores, [(0, 0, 0)])

    def test_three_pairs_share_one_question(self):
        a = torch.tensor([[-1., -2., -3.]], requires_grad=True)
        b = torch.tensor([[[-1., -2.], [-2., -1.], [-3., -1.]]], requires_grad=True)
        scores = {"a": a, "b": b, "a_ref": a.detach().clone(), "b_ref": b.detach().clone()}
        pairs = [(0, 0, 1), (0, 0, 2), (0, 1, 2)]
        losses = preference_losses(scores, pairs)
        self.assertEqual(losses["N"], 1)
        self.assertEqual(losses["M"], 3)
        self.assertAlmostEqual(losses["L_A_weighted"].item(), 0.5 * math.log(2), places=6)
        losses["L_A_weighted"].backward()
        self.assertLess(a.grad[0, 0].item(), 0)
        self.assertAlmostEqual(a.grad[0, 1].item(), 0)
        self.assertGreater(a.grad[0, 2].item(), 0)
        self.assertIsNone(b.grad)


if __name__ == "__main__":
    unittest.main()
