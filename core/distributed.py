"""Initialize and close the process group created by torchrun."""
from contextlib import contextmanager
from datetime import timedelta
import os

import torch
import torch.distributed as dist


@contextmanager
def distributed_job(device="cuda"):
    parallel = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if parallel:
        if device == "cuda":
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        # DBS shards can finish far apart; allow the collection job's 30-hour limit.
        dist.init_process_group(backend="nccl" if device == "cuda" else "gloo",
                                timeout=timedelta(hours=30))
    try:
        yield
    finally:
        if parallel:
            dist.destroy_process_group()
