"""Teacher-forced completion scores and SysDPO-Sampling plus intermediate DPO.

DPO margin adapted from SysDPO multi_llm/train.py, commit
9e775895e7c01a7cd15276661fbc819f7ffc8c02. Copyright 2024 SysDPO
Contributors; adapted portions Apache-2.0 (LICENSE). Added log-space
marginalization, completion/EOS masking and independent intermediate loss.
"""

from dataclasses import asdict

import torch
import torch.nn.functional as F

from core.schemas import ScoringRequest


LAMBDA_A = 0.5


def dpo_pair_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.5):
    main_inside = chosen - rejected - ref_chosen.detach() + ref_rejected.detach()
    main_loss = -F.logsigmoid(beta * main_inside)
    return main_loss


def get_probs(logits, seq, attn, reduction="mean"):
    """Upstream-style shifted completion mask; return score and log score."""
    if reduction not in ("sum", "mean"):
        raise ValueError("reduction must be sum or mean")
    if logits.ndim != 3 or seq.shape != logits.shape[:2] or attn.shape != seq.shape:
        raise ValueError("Expected logits [batch,time,vocab] and matching seq/attn")

    seq = seq.to(torch.int64)
    mask = attn[:, 1:].to(torch.bool)
    if not mask.any(dim=1).all():
        raise ValueError("Every sequence needs at least one scored completion token")

    # Masking invalid logits later can hide NaNs in the score but not its gradient.
    logits = logits.float()
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite model logits")

    logits = logits[:, :-1, :].contiguous()
    token_log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = token_log_probs.gather(-1, seq[:, 1:, None]).squeeze(-1)
    log_probs = torch.where(mask, token_log_probs, 0).sum(dim=-1)
    if reduction == "mean":
        log_probs = log_probs / mask.sum(dim=-1)

    if not torch.isfinite(log_probs).all():
        raise FloatingPointError("Non-finite completion log scores")
    return torch.exp(log_probs), log_probs


def completion_logp(model, tokenizer, messages, completion, reduction="mean"):
    """Score completion and EOS; exclude prompt tokens and never truncate input."""
    if reduction not in ("sum", "mean"):
        raise ValueError("reduction must be sum or mean")

    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    suffix = tokenizer.encode(completion, add_special_tokens=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define EOS")
    if not suffix or suffix[-1] != tokenizer.eos_token_id:
        suffix.append(tokenizer.eos_token_id)
    if not prefix:
        raise ValueError("Chat template returned an empty prefix")

    device = next(model.parameters()).device
    tokens = torch.tensor([prefix + suffix], dtype=torch.long, device=device)
    attn = torch.cat([torch.zeros(1, len(prefix), device=device),
                      torch.ones(1, len(suffix), device=device)], dim=1)

    output = model(input_ids=tokens, attention_mask=torch.ones_like(tokens))
    logits = output.logits if hasattr(output, "logits") else output["logits"]
    _, log_probs = get_probs(logits, tokens, attn, reduction)
    return log_probs[0]


def score_request(request: ScoringRequest, mid_model, output_model, ref_mid_model, ref_output_model,
                  mid_tokenizer, output_tokenizer, reduction="mean"):
    """Score both final answers under every candidate context, plus A and refs."""
    request = ScoringRequest(**asdict(request))

    def score_models(mid_model, output_model):
        log_mid_probs = torch.stack([
            completion_logp(mid_model, mid_tokenizer, request.a_messages, y, reduction)
            for y in request.intermediates])
        log_final_probs = torch.stack([torch.stack([
            completion_logp(output_model, output_tokenizer, messages, z, reduction)
            for z in request.final_answers
        ]) for messages in request.b_messages])
        return log_mid_probs, log_final_probs

    current_a, current_b = score_models(mid_model, output_model)
    with torch.no_grad():
        reference_a, reference_b = score_models(ref_mid_model, ref_output_model)
    return {"a": current_a, "b": current_b, "a_ref": reference_a, "b_ref": reference_b}


def preference_losses(scores, pairs, candidate_mask=None, beta_sys=0.5, beta_a=0.5):
    """A [N,K], B [N,K,2]; pairs: (row, chosen_index, rejected_index)."""
    a, b, a_ref, b_ref = (scores[key] for key in ("a", "b", "a_ref", "b_ref"))
    if a.ndim != 2:
        raise ValueError("Expected A score matrix [N,K]")

    n, k = a.shape
    if n == 0 or k < 2 or a_ref.shape != a.shape or b.shape != (n, k, 2) or b_ref.shape != b.shape:
        raise ValueError("Expected nonempty A [N,K] and B [N,K,2] score tensors")
    if any(not torch.isfinite(t).all() or (t > 1e-6).any() for t in (a, b, a_ref, b_ref)):
        raise ValueError("Completion log scores must be finite and nonpositive")

    if candidate_mask is None:
        candidate_mask = torch.ones_like(a, dtype=torch.bool)
    if candidate_mask.shape != a.shape or candidate_mask.dtype != torch.bool or not candidate_mask.any(dim=1).all():
        raise ValueError("Each question needs a nonempty boolean candidate mask")

    policy = torch.logsumexp((a[:, :, None] + b).masked_fill(~candidate_mask[:, :, None], -torch.inf), dim=1)
    reference = torch.logsumexp((a_ref.detach()[:, :, None] + b_ref.detach()).masked_fill(
        ~candidate_mask[:, :, None], -torch.inf), dim=1)
    system_rows = dpo_pair_loss(policy[:, 0], policy[:, 1], reference[:, 0], reference[:, 1], beta_sys)

    auxiliary = a.new_zeros(())
    if pairs:
        indices = [pair[0] for pair in pairs]
        if any(not 0 <= i < n for i in indices):
            raise ValueError("Intermediate pairs need valid question indices")
        if any(not 0 <= p[1] < k or not 0 <= p[2] < k or p[1] == p[2] for p in pairs):
            raise ValueError("Intermediate pairs need distinct valid candidate indices")

        rows = torch.tensor(indices, device=a.device)
        chosen = torch.tensor([pair[1] for pair in pairs], device=a.device)
        rejected = torch.tensor([pair[2] for pair in pairs], device=a.device)
        if not (candidate_mask[rows, chosen] & candidate_mask[rows, rejected]).all():
            raise ValueError("Intermediate preference requires two distinct candidates")

        losses = dpo_pair_loss(a[rows, chosen], a[rows, rejected],
                              a_ref[rows, chosen], a_ref[rows, rejected], beta_a)
        auxiliary = LAMBDA_A * losses.mean()

    system = system_rows.mean()
    return {"L_sys": system, "L_A_weighted": auxiliary, "L_total": system + auxiliary,
            "final_reward_w": (policy[:, 0] - reference[:, 0]).mean(),
            "final_reward_l": (policy[:, 1] - reference[:, 1]).mean(),
            "N": n, "M": len(pairs)}
