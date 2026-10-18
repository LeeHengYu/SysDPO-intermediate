"""Offline LoRA updates for saved SysDPO requests; no network calls."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import random
import shutil

from pydantic import TypeAdapter
from pydantic.dataclasses import dataclass
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, get_constant_schedule_with_warmup
from peft import LoraConfig, PeftModel, get_peft_model

from core.likelihood import completion_logp, preference_losses, score_request
from core.schemas import IntermediatePreference, ScoringRequest
from core.workflow import save_json


@dataclass
class TrainingConfig:
    optimizer_rows: int = 64
    checkpoint_rows: int = 256
    learning_rate: float = 6.4e-6
    warmup_steps: int = 20
    beta: float = 0.5
    seed: int = 42
    rank: int = 16
    alpha: int = 16
    dropout: float = 0.05
    reduction: str = "mean"

    def __post_init__(self):
        if self.optimizer_rows < 1 or self.checkpoint_rows < 1:
            raise ValueError("Row counts must be positive")
        if self.checkpoint_rows % self.optimizer_rows:
            raise ValueError("Optimizer rows must divide checkpoint rows")
        if self.learning_rate <= 0 or self.warmup_steps < 0:
            raise ValueError("Invalid learning rate or warmup")
        if self.beta <= 0 or self.rank < 1 or self.alpha < 1 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid DPO or adapter parameters")
        if self.reduction not in ("mean", "sum"):
            raise ValueError("Unknown completion reduction")


def load_models(base, config, checkpoint=None, device="cuda"):
    """Independent A/B adapters; one immutable base suffices for both references."""
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    models = []
    for role in ("A", "B", "reference"):
        model = AutoModelForCausalLM.from_pretrained(
            base, local_files_only=True, torch_dtype=dtype, attn_implementation="sdpa")
        model.config.use_cache = False
        if role == "reference":
            model.eval().requires_grad_(False)
        elif checkpoint:
            model = PeftModel.from_pretrained(model, checkpoint / f"{role}_adapter", is_trainable=True)
        else:
            model = get_peft_model(model, LoraConfig(
                r=config.rank, lora_alpha=config.alpha, lora_dropout=config.dropout,
                bias="none", task_type="CAUSAL_LM",
                target_modules=["k_proj", "gate_proj", "v_proj", "up_proj", "q_proj", "o_proj", "down_proj"]))
        if role != "reference":
            model.train()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        models.append(model.to(device))
    return (*models, tokenizer)


def read_training_rows(collection):
    requests = TypeAdapter(list[ScoringRequest]).validate_json((collection / "scoring_requests.json").read_text())
    preferences = TypeAdapter(list[IntermediatePreference]).validate_json(
        (collection / "intermediate_preferences.json").read_text())
    by_id = {row.id: row for row in requests}
    if not requests or len(by_id) != len(requests):
        raise ValueError("Training requires nonempty unique question IDs")
    pairs = {row.id: [] for row in requests}
    for pair in preferences:
        row = by_id[pair.id]
        if pair.a_messages != row.a_messages:
            raise ValueError("Intermediate preference has different A conditioning")
        edge = (0, row.intermediates.index(pair.y_chosen), row.intermediates.index(pair.y_rejected))
        if edge in pairs[row.id]:
            raise ValueError("Duplicate intermediate preference")
        pairs[row.id].append(edge)
    digest = hashlib.sha256()
    for name in ("scoring_requests.json", "intermediate_preferences.json"):
        digest.update((collection / name).read_bytes())
    return requests, pairs, digest.hexdigest()


def backward_row(request, pairs, models, tokenizer, system_count, pair_count, config):
    """Exact chain rule with one completion graph live at a time.

    Replay the same dropout RNG sequence so the score pass and gradient pass
    describe the same stochastic forward computation.
    """
    a, b, reference = models
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if next(a.parameters()).is_cuda else None
    with torch.no_grad():
        scores = score_request(request, a, b, reference, reference, tokenizer, tokenizer, config.reduction)
    final_cpu_rng = torch.get_rng_state()
    final_cuda_rng = torch.cuda.get_rng_state_all() if cuda_rng is not None else None
    leaves = {key: value.detach().unsqueeze(0) for key, value in scores.items()}
    leaves["a"].requires_grad_()
    leaves["b"].requires_grad_()
    mask = torch.tensor([request.candidate_mask], device=leaves["a"].device)
    losses = preference_losses(leaves, pairs, mask, config.beta, config.beta)
    loss = losses["L_sys"] / system_count
    if pair_count:
        loss = loss + losses["L_A_weighted"] * len(pairs) / pair_count
    da, db = torch.autograd.grad(loss, (leaves["a"], leaves["b"]))

    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    try:
        for i, y in enumerate(request.intermediates):
            score = completion_logp(a, tokenizer, request.a_messages, y, config.reduction)
            score.backward(da[0, i])
        for i, messages in enumerate(request.b_messages):
            for j, z in enumerate(request.final_answers):
                score = completion_logp(b, tokenizer, messages, z, config.reduction)
                score.backward(db[0, i, j])
    finally:
        torch.set_rng_state(final_cpu_rng)
        if final_cuda_rng is not None:
            torch.cuda.set_rng_state_all(final_cuda_rng)
    return float(loss.detach())


def copy_checkpoint(source, destination):
    """Publish only complete snapshots; interrupted copies are never resumable."""
    target = destination / source.name
    pending = destination / (source.name + ".partial")
    destination.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    shutil.copytree(source, pending, dirs_exist_ok=True)
    (pending / "complete").write_text("complete\n")
    pending.rename(target)
    return target


def save_checkpoint(models, optimizer, scheduler, config, state, directory, rank_rng_states=None):
    directory.mkdir(parents=True, exist_ok=False)
    for role, model in zip(("A", "B"), models):
        model.save_pretrained(directory / f"{role}_adapter")
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "rank_rng_states": rank_rng_states},
               directory / "training_state.pt")
    save_json(directory / "state.json", {**state, "config": asdict(config)})
    return directory


def restore_rng(state, rank, device):
    rng_states = state.get("rank_rng_states") or [state]
    if rank < len(rng_states):
        rng = rng_states[rank]
        torch.set_rng_state(rng["torch_rng"])
        random.setstate(rng["python_rng"])
        if device == "cuda" and rng["cuda_rng"]:
            # Older checkpoints used two visible GPUs on one node.
            source_device = rng.get("cuda_device", rank % len(rng["cuda_rng"]))
            torch.cuda.set_rng_state(rng["cuda_rng"][source_device])


def train_collection(base, collection, work, durable, config, resume=None, previous=None, device="cuda"):
    parallel = dist.is_initialized()
    rank = dist.get_rank() if parallel else 0
    world_size = dist.get_world_size() if parallel else 1
    if resume and previous:
        raise ValueError("Use resume for this collection or previous for a new collection")
    requests, pairs, data_hash = read_training_rows(collection)
    base_identity = json.loads((base / "provenance.json").read_text())
    checkpoint = resume or previous
    start, step = 0, 0
    if checkpoint:
        if not (checkpoint / "complete").exists():
            raise ValueError("Checkpoint has not completed synchronization")
        saved = json.loads((checkpoint / "state.json").read_text())
        if saved["config"] != asdict(config):
            raise ValueError("Training configuration differs from checkpoint")
        if saved["base_identity"] != base_identity:
            raise ValueError("Checkpoint was trained against different base/reference weights")
        if resume:
            if saved.get("world_size", 1) != world_size:
                raise ValueError("Resume must use the checkpoint's GPU/process count")
            if saved["data_hash"] != data_hash:
                raise ValueError("Resume data differs from checkpoint")
            start = saved["rows_done"]
        elif saved["rows_done"] != saved["total_rows"]:
            raise ValueError("Previous collection is incomplete")
        step = saved["optimizer_steps"]

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    a, b, reference, tokenizer = load_models(base, config, checkpoint, device)
    if device == "cuda" and parallel:
        torch.cuda.manual_seed(config.seed + rank)
    parameters = [p for model in (a, b) for p in model.parameters() if p.requires_grad]
    if parallel:
        for parameter in parameters:
            dist.broadcast(parameter.data, src=0)
    optimizer = torch.optim.RMSprop(parameters, lr=config.learning_rate)
    scheduler = get_constant_schedule_with_warmup(optimizer, config.warmup_steps)
    if checkpoint:
        state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng(state, rank, device)

    work.mkdir(parents=True, exist_ok=True)
    copies = []
    with ThreadPoolExecutor(max_workers=1) as worker:
        for offset in range(start, len(requests), config.optimizer_rows):
            for future in copies:
                if future.done():
                    future.result()
            batch = requests[offset:offset + config.optimizer_rows]
            pair_count = sum(len(pairs[row.id]) for row in batch)
            optimizer.zero_grad(set_to_none=True)
            loss = sum(backward_row(row, pairs[row.id], (a, b, reference), tokenizer,
                                    len(batch), pair_count, config) for row in batch[rank::world_size])
            if parallel:
                for parameter in parameters:
                    if parameter.grad is None:
                        parameter.grad = torch.zeros_like(parameter)
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                combined_loss = torch.tensor(loss, device=device, dtype=torch.float64)
                dist.all_reduce(combined_loss, op=dist.ReduceOp.SUM)
                loss = combined_loss.item()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            done = offset + len(batch)
            if rank == 0:
                print(f"rows={done}/{len(requests)} step={step} loss={loss:.6f} pairs={pair_count} workers={world_size}", flush=True)
            if done % config.checkpoint_rows == 0 or done == len(requests):
                rng = {"torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                       "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
                       "cuda_device": torch.cuda.current_device() if device == "cuda" else None}
                rank_rng_states = [None] * world_size if rank == 0 else None
                if parallel:
                    dist.gather_object(rng, rank_rng_states, dst=0)
                else:
                    rank_rng_states = [rng]
                if rank == 0:
                    snapshot = save_checkpoint((a, b), optimizer, scheduler, config,
                        {"rows_done": done, "total_rows": len(requests), "optimizer_steps": step,
                         "data_hash": data_hash, "base_identity": base_identity, "last_loss": loss,
                         "world_size": world_size}, work / f"rows_{done:06d}", rank_rng_states)
                    copies.append(worker.submit(copy_checkpoint, snapshot, durable))
        for future in copies:
            future.result()
    if parallel:
        dist.barrier()
    final = durable / f"rows_{len(requests):06d}"
    if not (final / "complete").exists():
        raise RuntimeError("Final durable checkpoint is missing")
    return final


