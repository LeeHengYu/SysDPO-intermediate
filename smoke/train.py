"""Procedural SysDPO-style collection, batching and update utilities."""

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from itertools import combinations, islice
import json
from pathlib import Path
import random

from datasets import load_dataset
from pydantic import TypeAdapter

from model_api import call_gen, call_judge, response_text
from likelihood import LAMBDA_A, compute_saved_losses
from topo_sort import topological_sort
from schemas import (
    DatasetRow, IntermediatePreference, Judgment, Prompts,
    QuestionRecord, ScoringRequest, SystemPreference,
)


GENERATOR = "qwen/qwen-2.5-7b-instruct"
JUDGE = "microsoft/microsoft-decision-1"
PROMPT_FILE = Path(__file__).with_name("prompts.json")


def freeze(model):
    return deepcopy(model).eval().requires_grad_(False)


def accumulate_gradients(loss_fn, batch, microbatch):
    """Sum microbatch losses / full batch count, including a short last chunk."""
    size = len(batch[0])
    if size == 0 or microbatch < 1 or any(len(column) != size for column in batch):
        raise ValueError("Need a nonempty aligned batch and positive microbatch size.")

    total = 0.0
    for start in range(0, size, microbatch):
        chunk = tuple(column[start:start + microbatch] for column in batch)
        contribution = loss_fn(chunk).sum() / size
        contribution.backward()
        total += contribution.detach().item()

    return total


def load_prompts(path) -> Prompts:
    return TypeAdapter(Prompts).validate_json(path.read_text())


def model_messages(row: DatasetRow, prompts: Prompts, instruction, draft=None):
    system = ("Dataset instruction:\n" + row.system + "\n\n"
              + prompts.instruction_policy + "\n\n" + instruction)

    content = row.question if draft is None else json.dumps(
        {"question": row.question, "draft": draft}, ensure_ascii=False
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(TypeAdapter(object).dump_python(value, mode="json"),
                                    indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_questions(output_dir, count) -> list[DatasetRow]:
    path = output_dir / "dataset.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if len(saved["rows"]) != count:
            raise ValueError("Cached question count differs; use another output directory")
        return [DatasetRow(**row) for row in saved["rows"]]

    dataset = load_dataset("Intel/orca_dpo_pairs", split="train", streaming=True)
    rows = [{"id": index, **row} for index, row in enumerate(islice(dataset, count))]
    if len(rows) != count:
        raise ValueError("Dataset returned the wrong number of questions")
    validated = [DatasetRow(**row) for row in rows]

    save_json(path, {"source": "Intel/orca_dpo_pairs", "split": "train",
                     "retrieved_utc": datetime.now(timezone.utc).isoformat(), "rows": rows})
    return validated


def recorded_call(output_dir, row_id, stage, model, messages, **settings):
    request = {"model": model, "messages": messages, **settings}
    digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    path = output_dir / "calls" / f"{row_id}_{stage}.json"

    if path.exists():
        record = json.loads(path.read_text())
        if record["request_sha256"] != digest:
            raise ValueError("Cached request changed; use another output directory")
        if "error" in record:
            raise RuntimeError("Previous call failed; no automatic retry")
        return response_text(record["response"], allow_truncated=not settings.get("json_output", False))

    record = {"row_id": row_id, "stage": stage, "request_sha256": digest,
              "request": request, "created_utc": datetime.now(timezone.utc).isoformat()}
    # Persist the request first: interruption does not silently trigger another paid call.
    save_json(path, {**record, "error": "Request started; response not yet saved"})

    try:
        if settings.get("json_output", False):
            record["response"] = call_judge(
                json.loads(messages[1]["content"]), messages[0]["content"], model=model)
        else:
            record["response"] = call_gen(model, messages, **settings)
    except Exception as error:
        record["error"] = type(error).__name__ + ": " + str(error)
        save_json(path, record)
        raise

    save_json(path, record)  # Includes raw model/judge output, provider and usage.
    return response_text(record["response"], allow_truncated=not settings.get("json_output", False))


def finish_reason(output_dir, row_id, stage):
    path = output_dir / "calls" / f"{row_id}_{stage}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["response"]["choices"][0]["finish_reason"]


def judge_pair(output_dir, row: DatasetRow, first, second, stage, first_id, second_id,
               prompts: Prompts) -> Judgment:
    order = [first_id, second_id]
    random.Random(f"7:{row.id}:{stage}").shuffle(order)
    candidates = {first_id: first, second_id: second}

    task = {"system_instruction": row.system, "question": row.question,
            "candidate_1": candidates[order[0]], "candidate_2": candidates[order[1]]}
    task["instruction_policy"] = prompts.instruction_policy
    task["response_role"] = prompts.a_instruction
    task["comparison_type"] = "intermediate"

    raw = recorded_call(output_dir, row.id, stage, JUDGE, [
        {"role": "system", "content": prompts.judge_rubric},
        {"role": "user", "content": json.dumps(task, ensure_ascii=False)},
    ], temperature=0, max_tokens=512, json_output=True)

    decision = TypeAdapter(Judgment).validate_json(raw)
    decision.presented_order = order
    decision.winner = order[0] if decision.choice == "option1" else order[1] if decision.choice == "option2" else None
    return decision


def process_question(output_dir, row: DatasetRow, prompts: Prompts, max_tokens) -> QuestionRecord:
    prompt = model_messages(row, prompts, prompts.a_instruction)
    record = QuestionRecord(id=row.id, x=row.question, system=row.system,
                            z_chosen=row.chosen, z_rejected=row.rejected, a_messages=prompt)

    for name in ("y1", "y2", "y3"):
        try:
            setattr(record, name, recorded_call(output_dir, row.id, name, GENERATOR, prompt,
                                                temperature=0.3, max_tokens=max_tokens))
            record.finish_reasons[name] = finish_reason(output_dir, row.id, name)
            if record.finish_reasons[name] != "stop":
                raise ValueError("Incomplete generation retained for inspection only")
        except Exception as error:
            record.errors.append({"stage": name, "error": type(error).__name__ + ": " + str(error)})

    candidates = {"y1": record.y1, "y2": record.y2, "y3": record.y3}
    complete = all(record.finish_reasons.get(name) == "stop" for name in candidates)
    if complete:
        for first, second in combinations(candidates, 2):
            stage = f"judge_{first}_{second}"
            try:
                if candidates[first].strip() == candidates[second].strip():
                    record.discard_reason = "duplicate"
                    decision = Judgment(choice="tie")
                else:
                    decision = judge_pair(output_dir, row, candidates[first], candidates[second],
                                          stage, first, second, prompts)
                record.intermediate_judgments.append(decision)
                if not decision.winner:
                    record.discard_reason = record.discard_reason or decision.choice
                    break
                loser = second if decision.winner == first else first
                record.preference_edges.append((decision.winner, loser))
            except Exception as error:
                record.errors.append({"stage": stage, "error": type(error).__name__ + ": " + str(error)})
                break

        if len(record.preference_edges) == 3:
            record.comparisons_accepted = topological_sort(record.preference_edges)
            if not record.comparisons_accepted:
                record.discard_reason = "cyclic preferences"
        else:
            record.discard_reason = record.discard_reason or "incomplete strict comparisons"

    # No final generation or final judge: B teacher-forces dataset zc/zr instead.
    if complete:
        record.scoring_request = ScoringRequest(
            id=row.id, a_messages=prompt, intermediates=list(candidates.values()),
            b_messages=[model_messages(row, prompts, prompts.b_instruction, y) for y in candidates.values()],
            final_answers=[row.chosen, row.rejected],
            candidate_mask=[y not in list(candidates.values())[:i] for i, y in enumerate(candidates.values())])

    save_json(output_dir / f"question_{row.id}.json", record)
    return record


def update_models(system_batch: list[SystemPreference], intermediate_batch: list[IntermediatePreference]):
    """Validate collected inputs only; no likelihood scoring/backprop/weight updates."""
    if any(not isinstance(row, SystemPreference) for row in system_batch) or any(
        not isinstance(row, IntermediatePreference) for row in intermediate_batch
    ):
        raise ValueError("Expected validated system and intermediate preference rows")
    if any(row.y_chosen == row.y_rejected for row in intermediate_batch):
        raise ValueError("Intermediate preferences need distinct candidates")

    return {"status": "weight_update_abstracted", "system_rows": len(system_batch),
            "intermediate_rows": len(intermediate_batch), "optimizer_steps": 0}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=int, default=10)
    parser.add_argument("--scores", type=Path, help="Externally teacher-forced current/reference score JSON")
    parser.add_argument("--prompt-file", type=Path, default=PROMPT_FILE)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "runs" / "batch10_sampling")
    args = parser.parse_args()

    if not 1 <= args.questions <= 100:
        parser.error("questions must be between 1 and 100")
    if args.max_tokens < 1:
        parser.error("max-tokens must be positive")

    return args


def main():
    args = parse_args()
    prompts = load_prompts(args.prompt_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "calls").mkdir(exist_ok=True)

    config = {"workflow_version": 11, "questions": args.questions, "max_tokens": args.max_tokens,
              "generation_temperature": 0.3,
              "lambda_A": LAMBDA_A,
              "prompts": asdict(prompts), "generator": GENERATOR, "judge": JUDGE}
    config_path = args.output_dir / "run_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise SystemExit("Run configuration changed; use another output directory")
    elif any((args.output_dir / "calls").iterdir()):
        raise SystemExit("Existing calls have no compatible run configuration; use another output directory")
    save_json(config_path, config)

    rows = load_questions(args.output_dir, args.questions)
    system_batch = [SystemPreference(id=row.id, x=row.question, system=row.system,
                                     z_chosen=row.chosen, z_rejected=row.rejected) for row in rows]
    save_json(args.output_dir / "system_preferences.json", system_batch)

    print("***** Running SysDPO-Sampling collection smoke *****", flush=True)
    print(f"  Num examples = {len(rows)}; Collection rounds = 1; Intermediate samples = 3", flush=True)
    print("  Final supervision = dataset chosen/rejected; Optimizer steps = 0", flush=True)
    print(f"  beta_sys=0.5 beta_A=0.5; lambda_A={LAMBDA_A}", flush=True)

    results, errors = [], []
    for row in rows:
        try:
            result = process_question(args.output_dir, row, prompts, args.max_tokens)
            results.append(result)
            errors.extend({"id": row.id, **error} for error in result.errors)
            print(f"  step={len(results):02d}/{len(rows)} question={row.id} "
                  f"comparisons_accepted={result.comparisons_accepted} "
                  f"scoring_request={'ready' if result.scoring_request else 'unavailable'} "
                  f"errors={len(result.errors)}", flush=True)
        except Exception as error:
            errors.append({"id": row.id, "error": type(error).__name__ + ": " + str(error)})
            print(f"question {row.id}: failed ({type(error).__name__})", flush=True)

    intermediate = []
    for row in results:
        if not row.comparisons_accepted:
            continue
        candidates = {"y1": row.y1, "y2": row.y2, "y3": row.y3}
        for winner, loser in row.preference_edges:
            intermediate.append(IntermediatePreference(
                id=row.id, x=row.x, system=row.system, a_messages=row.a_messages,
                y_chosen=candidates[winner], y_rejected=candidates[loser],
                finish_reasons=row.finish_reasons))

    save_json(args.output_dir / "intermediate_preferences.json", intermediate)

    requests = [row.scoring_request for row in results if row.scoring_request is not None]
    save_json(args.output_dir / "scoring_requests.json", requests)

    metrics = {
        "status": "likelihood_backend_pending", "N": len(requests), "M": len(intermediate),
        "L_sys": None, "L_A_weighted": None, "L_total": None,
        "final_reward_w": None, "final_reward_l": None,
    }
    if args.scores:
        try:
            metrics = compute_saved_losses(args.scores, requests, intermediate)
        except Exception as error:
            failure = type(error).__name__ + ": " + str(error)
            metrics.update(status="loss_evaluation_failed", error=failure)
            errors.append({"stage": "scoring", "error": failure})
            print(f"  Scoring failed: {failure}", flush=True)

    save_json(args.output_dir / "losses.json", metrics)

    totals = Counter()
    stages = Counter()
    for path in (args.output_dir / "calls").glob("*.json"):
        call = json.loads(path.read_text())
        stages[call["stage"]] += 1
        usage = call.get("response", {}).get("usage", {})
        for name, value in usage.items():
            if type(value) in (int, float):
                totals[{"input_tokens": "prompt_tokens", "output_tokens": "completion_tokens"}.get(name, name)] += value
        if "input_tokens" in usage and "output_tokens" in usage and "total_tokens" not in usage:
            totals["total_tokens"] += usage["input_tokens"] + usage["output_tokens"]

    summary = {"requested_questions": args.questions,
               "completed_questions": sum(not row.errors for row in results),
               "collection_rounds": 1, "generator": GENERATOR, "receiver": GENERATOR,
               "intermediate_judge": JUDGE, "final_judge": None, "losses": metrics,
               "calls_by_stage": dict(stages), "usage": dict(totals), "errors": errors,
               "update": update_models(system_batch, intermediate)}
    save_json(args.output_dir / "summary.json", summary)

    print(f"  N_scoring_ready={metrics['N']} M_intermediate={metrics['M']}", flush=True)
    print("  " + " ".join(f"{key}={metrics[key]:.6f}" if metrics[key] is not None else f"{key}=pending"
                          for key in ("L_sys", "L_A_weighted", "L_total", "final_reward_w", "final_reward_l")), flush=True)
    print(f"  Tokens: input={totals['prompt_tokens']} output={totals['completion_tokens']} "
          f"total={totals['total_tokens']}; status={metrics['status']}", flush=True)

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
