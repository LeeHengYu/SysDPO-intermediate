"""Generate a frozen A snapshot's three DBS candidates without network access."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from core.schemas import DatasetRow, QuestionRecord
from core.distributed import distributed_job
from core.workflow import PROMPT_FILE, load_prompts, model_messages, save_json


def collect(base, data, output, publish, offset=0, count=None, checkpoint=None,
            prompt_file=PROMPT_FILE, max_tokens=512, device="cuda"):
    parallel = dist.is_initialized()
    rank = dist.get_rank() if parallel else 0
    world_size = dist.get_world_size() if parallel else 1

    prompts = load_prompts(prompt_file)
    stop = None if count is None else offset + count
    rows = [DatasetRow(**row) for row in json.loads(data.read_text())[offset:stop]]
    if not rows:
        raise ValueError("No collection rows selected")

    identity = json.loads((base / "provenance.json").read_text())
    snapshot = json.loads((checkpoint / "state.json").read_text()) if checkpoint else None
    if checkpoint and (not (checkpoint / "complete").exists() or snapshot["base_identity"] != identity):
        raise ValueError("Invalid collection checkpoint")

    config = {"base": identity, "snapshot": snapshot, "prompts": asdict(prompts),
              "rows": [asdict(row) for row in rows], "max_tokens": max_tokens,
              "num_beams": 3, "num_beam_groups": 3, "diversity_penalty": 10.0}

    output.mkdir(parents=True, exist_ok=True)
    publish.mkdir(parents=True, exist_ok=True)
    config_path = publish / "collection_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Collection configuration changed")

    if rank == 0:
        save_json(output / "collection_config.json", config)
        save_json(output / "dataset.json", {"rows": [asdict(row) for row in rows]})
        for name in ("collection_config.json", "dataset.json"):
            shutil.copy2(output / name, publish / (name + ".partial"))
            (publish / (name + ".partial")).replace(publish / name)

    if parallel:
        dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32, attn_implementation="sdpa")
    if checkpoint:
        model = PeftModel.from_pretrained(model, checkpoint / "A_adapter")

    model.to(device).eval().requires_grad_(False)

    for row in rows[rank::world_size]:
        name = f"question_{row.id}.json"
        if (publish / name).exists():
            print(f"question={row.id} cached", flush=True)
            continue

        messages = model_messages(row, prompts, prompts.a_instruction)
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_tensors="pt").to(device)

        with torch.inference_mode():
            sequences = model.generate(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                num_beams=3, num_beam_groups=3, num_return_sequences=3, diversity_penalty=10.0,
                do_sample=False, temperature=1.0, top_p=1.0, max_new_tokens=max_tokens,
                pad_token_id=tokenizer.eos_token_id)

        record = QuestionRecord(id=row.id, x=row.question, system=row.system,
            z_chosen=row.chosen, z_rejected=row.rejected, a_messages=messages)

        counts = {}
        for name_y, sequence in zip(("y1", "y2", "y3"), sequences):
            completion = sequence[tokens.shape[1]:].tolist()
            finished = tokenizer.eos_token_id in completion
            if finished:
                completion = completion[:completion.index(tokenizer.eos_token_id) + 1]
            setattr(record, name_y, tokenizer.decode(completion, skip_special_tokens=True))
            record.finish_reasons[name_y] = "stop" if finished else "length"
            counts[name_y] = len(completion)

        save_json(output / name, record)
        save_json(output / f"tokens_{row.id}.json", {"prompt_tokens": tokens.shape[1], "completion_tokens": counts})
        for filename in (f"tokens_{row.id}.json", name):
            shutil.copy2(output / filename, publish / (filename + ".partial"))
            (publish / (filename + ".partial")).replace(publish / filename)

        print(f"question={row.id} finishes={record.finish_reasons} tokens={counts}", flush=True)

    if parallel:
        dist.barrier()

    if rank == 0:
        (publish / "candidates.ready").write_text("complete\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "data", "output", "publish"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, help="Rows to collect; defaults to all remaining input rows")
    parser.add_argument("--max-tokens", type=int, default=512)

    args = parser.parse_args()

    with distributed_job():
        collect(**vars(args))
