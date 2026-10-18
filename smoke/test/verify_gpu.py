"""Full-size GPU recovery check against an existing completed pilot collection."""
import argparse
import gc
import json
import os
from pathlib import Path

from safetensors.torch import load_file
import torch
import torch.distributed as dist
from torch.nn.attention import SDPBackend, sdpa_kernel

from core.likelihood import completion_logp
from core.training import TrainingConfig, load_models, read_training_rows, train_collection
from core.distributed import distributed_job
from core.workflow import save_json
from multi_llm.collect import collect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "collection", "work", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    rank = dist.get_rank() if dist.is_initialized() else 0
    args.output.mkdir(parents=True, exist_ok=True)
    requests, _, _ = read_training_rows(args.collection)
    checkpoints = sorted((args.collection / "checkpoints").glob("rows_*/complete"))
    original, intermediate = checkpoints[-1].parent, checkpoints[-2].parent
    config = TrainingConfig(**json.loads((original / "state.json").read_text())["config"])

    # Compare controlled runs; CUDA's default attention dispatch need not be bit-exact.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    with sdpa_kernel(SDPBackend.MATH):
        full = train_collection(args.base, args.collection, args.work / "full",
            args.output / "full", config)
        intermediate = args.output / "full" / intermediate.name
        gc.collect()
        torch.cuda.empty_cache()
        resumed = train_collection(args.base, args.collection, args.work / "resume",
            args.output / "resumed", config, resume=intermediate)
    gc.collect()
    torch.cuda.empty_cache()
    differences = {}
    for role in ("A", "B"):
        expected = load_file(str(full / f"{role}_adapter/adapter_model.safetensors"))
        actual = load_file(str(resumed / f"{role}_adapter/adapter_model.safetensors"))
        differences[role] = max((expected[name] - actual[name]).abs().max().item() for name in expected)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
        assert any(torch.count_nonzero(value) for name, value in actual.items() if "lora_B" in name)
    del expected, actual

    scores = []
    for _ in range(2):
        a, b, reference, tokenizer = load_models(args.base, config, resumed)
        a.eval()
        b.eval()
        assert not any(parameter.requires_grad for parameter in reference.parameters())
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            request = requests[0]
            scores.append(torch.stack([
                completion_logp(a, tokenizer, request.a_messages, request.intermediates[0]),
                completion_logp(b, tokenizer, request.b_messages[0], request.final_answers[0]),
                completion_logp(reference, tokenizer, request.a_messages, request.intermediates[0]),
            ]).cpu())
        del a, b, reference, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    torch.testing.assert_close(scores[0], scores[1], atol=1e-6, rtol=0)

    rows = json.loads((args.collection / "dataset.json").read_text())["rows"]
    if rank == 0:
        save_json(args.output / "next_rows.json", [row for row in rows if row["id"] == requests[0].id])
    if dist.is_initialized():
        dist.barrier()
    collect(args.base, args.output / "next_rows.json", args.work / "next_generated",
            args.output / "next_collection", count=1, checkpoint=resumed)
    if rank == 0:
        save_json(args.output / "result.json", {"status": "passed", "training_backend": "deterministic SDPA math",
            "workers": dist.get_world_size() if dist.is_initialized() else 1, "max_adapter_difference": differences,
            "reload_log_scores": [score.tolist() for score in scores], "resume_checkpoint": str(intermediate),
            "restored_checkpoint": str(resumed), "next_collection": str(args.output / "next_collection")})
        print("PASS: full-size A/B resume, repeated reload scores, frozen reference and next-snapshot generation", flush=True)


if __name__ == "__main__":
    with distributed_job():
        main()
