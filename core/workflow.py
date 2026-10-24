"""Shared prompts, durable API records and stateless pair judging."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

from pydantic import TypeAdapter
from core.model_api import call_judge, response_text
from core.schemas import DatasetRow, Judgment, Prompts

JUDGE = "microsoft/microsoft-decision-1"
PROMPT_FILE = Path(__file__).with_name("prompts.json")


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


def recorded_call(output_dir, row_id, stage, model, messages):
    request = {"model": model, "messages": messages}
    digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    path = output_dir / "calls" / f"{row_id}_{stage}.json"

    if path.exists():
        record = json.loads(path.read_text())
        # Older records include generation settings that Decisions never used.
        if {key: record["request"][key] for key in request} != request:
            raise ValueError("Cached request changed; use another output directory")
        if "error" in record:
            raise RuntimeError("Previous call failed; no automatic retry")
        return response_text(record["response"])

    record = {"row_id": row_id, "stage": stage, "request_sha256": digest,
              "request": request, "created_utc": datetime.now(timezone.utc).isoformat()}
    # Persist the request first: interruption does not silently trigger another paid call.
    save_json(path, {**record, "error": "Request started; response not yet saved"})

    try:
        record["response"] = call_judge(
            json.loads(messages[1]["content"]), messages[0]["content"], model=model)
        text = response_text(record["response"])
    except Exception as error:
        record["error"] = type(error).__name__ + ": " + str(error)
        save_json(path, record)
        raise

    save_json(path, record)  # Includes raw model/judge output, provider and usage.
    return text


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
    ])

    decision = TypeAdapter(Judgment).validate_json(raw)
    decision.presented_order = order
    decision.winner = order[0] if decision.choice == "option1" else order[1] if decision.choice == "option2" else None
    return decision


