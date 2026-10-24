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
        group_device = None
        if device == "cuda":
            local_device = 0 if torch.cuda.device_count() == 1 else int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_device)
            group_device = torch.device("cuda", local_device)

        # DBS shards can finish far apart; allow the collection job's 30-hour limit.
        # Bind NCCL before model loading so the first barrier cannot guess another GPU.
        dist.init_process_group(backend="nccl" if device == "cuda" else "gloo",
                                timeout=timedelta(hours=30), device_id=group_device)
    try:
        yield
    finally:
        if parallel:
            dist.destroy_process_group()
