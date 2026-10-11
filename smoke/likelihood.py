"""Teacher-forced completion scores and SysDPO-Sampling plus intermediate DPO.

DPO margin adapted from SysDPO multi_llm/train.py, commit
9e775895e7c01a7cd15276661fbc819f7ffc8c02. Copyright 2024 SysDPO
Contributors; adapted portions Apache-2.0 (LICENSE). Added log-space
marginalization, completion/EOS masking and independent intermediate loss.
"""

from dataclasses import asdict
import hashlib
import json

import torch
import torch.nn.functional as F

from schemas import IntermediatePreference, ScoringRequest


def confidence_weight(confidence):
    if not torch.isfinite(confidence).all() or not ((confidence >= 0) & (confidence <= 1)).all():
        raise ValueError("Confidence must be finite and in [0, 1].")

    # Placeholder f; the research mapping is still undecided.
    return 0.5 * confidence.detach()


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
    """All four supplied z/context combinations, plus both A candidates and refs."""
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
    """scores: A [N,2], B [N,2,2]; pairs: (row, chosen_y_index, confidence)."""
    a, b, a_ref, b_ref = (scores[key] for key in ("a", "b", "a_ref", "b_ref"))
    if a.ndim != 2:
        raise ValueError("Expected A score matrix [N,2]")

    n = a.shape[0]
    if n == 0 or a.shape != (n, 2) or a_ref.shape != a.shape or b.shape != (n, 2, 2) or b_ref.shape != b.shape:
        raise ValueError("Expected nonempty A [N,2] and B [N,2,2] score tensors")
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
        if len(set(indices)) != len(indices) or any(not 0 <= i < n for i in indices):
            raise ValueError("Intermediate pairs need distinct valid question indices")
        if any(pair[1] not in (0, 1) for pair in pairs):
            raise ValueError("Intermediate chosen index must be 0 or 1")

        rows = torch.tensor(indices, device=a.device)
        chosen = torch.tensor([pair[1] for pair in pairs], device=a.device)
        if not (candidate_mask[rows].all(dim=1)).all():
            raise ValueError("Intermediate preference requires two distinct candidates")

        confidence = a.new_tensor([pair[2] for pair in pairs])
        losses = dpo_pair_loss(a[rows, chosen], a[rows, 1 - chosen],
                              a_ref[rows, chosen], a_ref[rows, 1 - chosen], beta_a)
        auxiliary = (confidence_weight(confidence) * losses).sum() / len(pairs)

    system = system_rows.mean()
    return {"L_sys": system, "L_A_weighted": auxiliary, "L_total": system + auxiliary,
            "final_reward_w": (policy[:, 0] - reference[:, 0]).mean(),
            "final_reward_l": (policy[:, 1] - reference[:, 1]).mean(),
            "N": n, "M": len(pairs)}


def compute_saved_losses(path, requests: list[ScoringRequest], intermediate: list[IntermediatePreference]):
    """Evaluate actual supplied scores, never substitute judge scores for logp."""
    requests = [ScoringRequest(**asdict(request)) for request in requests]

    source = json.loads(path.read_text())
    if source.get("reduction") not in ("sum", "mean") or source.get("include_eos") is not True:
        raise ValueError("Score file must declare reduction=sum/mean and include_eos=true")

    rows = source["rows"]
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != {r.id for r in requests}:
        raise ValueError("Score rows must exactly match the scoring-ready question IDs")
    for request in requests:
        if by_id[request.id].get("request_sha256") != request.request_sha256:
            raise ValueError("Scores do not match exact prompts/completions")

    scores = {key: torch.tensor([by_id[r.id][key] for r in requests], dtype=torch.float64)
              for key in ("a", "b", "a_ref", "b_ref")}
    indices = {r.id: i for i, r in enumerate(requests)}
    pairs = []
    for pair in intermediate:
        i = indices[pair.id]
        chosen = requests[i].intermediates.index(pair.y_chosen)
        if pair.y_rejected != requests[i].intermediates[1 - chosen]:
            raise ValueError("Rejected intermediate does not match the scoring request")
        pairs.append((i, chosen, pair.confidence))

    mask = torch.tensor([r.candidate_mask for r in requests], dtype=torch.bool)
    losses = preference_losses(scores, pairs, mask)
    return {**{key: value.item() if torch.is_tensor(value) else value for key, value in losses.items()},
            "status": "loss_evaluated_no_weight_updates", "reduction": source["reduction"],
            "include_eos": True, "scores_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "confidence_weight": "placeholder f(c)=0.5*c"}
