from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from rubic_rl.datasets import read_jsonl
from rubic_rl.policies import TrainingConfig, train_policy


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a NumPy softmax policy on supervised Rubik JSONL data."
    )
    parser.add_argument("--dataset", required=True, type=Path, help="Input JSONL path.")
    parser.add_argument(
        "--model-out",
        type=Path,
        default=None,
        help="Optional output .npz checkpoint path.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    records = read_jsonl(args.dataset)
    model, result = train_policy(
        records,
        TrainingConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            l2=args.l2,
            validation_split=args.validation_split,
            seed=args.seed,
        ),
    )

    output = result.to_dict()
    if args.model_out is not None:
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        with args.model_out.open("wb") as output_file:
            model.save(output_file)
        output["model_out"] = str(args.model_out)

    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
