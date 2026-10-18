"""Train offline A/B adapters from a judged collection."""

import argparse
import json
from pathlib import Path

from core.training import TrainingConfig, train_collection
from core.distributed import distributed_job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")

    args = parser.parse_args()
    config = TrainingConfig(**json.loads(args.config.read_text())) if args.config else TrainingConfig()

    with distributed_job(args.device):
        final = train_collection(args.base, args.collection, args.work, args.checkpoints,
                                 config, args.resume, args.previous, args.device)

    print(f"Training complete: {final}", flush=True)


if __name__ == "__main__":
    main()
