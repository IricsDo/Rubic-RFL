"""Evaluate a DAVI value checkpoint with Weighted A* search, by scramble depth.

This is the promotion gate for the value-iteration track (the analogue of
``search_eval`` for the policy track). It reports solve rate, solution length,
and search cost per scramble depth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Sequence

import numpy as np

from rubic_rl import cube_ops as C
from rubic_rl.models.policy_value import load_policy_value_checkpoint
from rubic_rl.search import make_action_policy_fn, make_value_fn, weighted_astar_solve


def _parse_depths(values: Sequence[str]) -> tuple[int, ...]:
    raw: list[str] = []
    for value in values:
        raw.extend(part for part in value.split(",") if part)
    return tuple(int(depth) for depth in raw)


def evaluate(
    model,
    device,
    *,
    depths: tuple[int, ...],
    samples_per_depth: int,
    weight: float,
    policy_weight: float,
    batch_expansion: int,
    max_nodes: int,
    seed: int | None,
) -> dict:
    value_fn = make_value_fn(model, device)
    action_policy_fn = make_action_policy_fn(model, device) if policy_weight > 0.0 else None
    rng = np.random.default_rng(seed)
    by_depth: dict[str, dict] = {}
    all_solved: list[bool] = []
    all_len: list[int] = []
    all_ms: list[float] = []

    for depth in depths:
        if depth > 0:
            states, _ = C.scramble_batch(
                samples_per_depth, depth, rng, min_depth=depth
            )
        else:
            states = C.solved_batch(samples_per_depth)
        solved_flags: list[bool] = []
        lengths: list[int] = []
        expanded: list[int] = []
        durations: list[float] = []
        for row in states:
            start = time.perf_counter()
            result = weighted_astar_solve(
                row,
                value_fn,
                action_policy_fn=action_policy_fn,
                weight=weight,
                policy_weight=policy_weight,
                batch_expansion=batch_expansion,
                max_nodes=max_nodes,
            )
            durations.append((time.perf_counter() - start) * 1000.0)
            solved_flags.append(bool(result["solved"]))
            expanded.append(int(result["expanded"]))
            if result["solved"]:
                lengths.append(int(result["length"]))
        solve_rate = float(np.mean(solved_flags)) if solved_flags else 0.0
        by_depth[str(depth)] = {
            "samples": samples_per_depth,
            "solve_rate": solve_rate,
            "avg_len_solved": float(np.mean(lengths)) if lengths else None,
            "avg_expanded": float(np.mean(expanded)) if expanded else None,
            "avg_ms": float(np.mean(durations)) if durations else None,
        }
        all_solved.extend(solved_flags)
        all_len.extend(lengths)
        all_ms.extend(durations)
        print(
            f"[davi-eval] depth {depth:>2}: solve_rate={solve_rate:.3f} "
            f"avg_len={by_depth[str(depth)]['avg_len_solved']} "
            f"avg_ms={by_depth[str(depth)]['avg_ms']:.0f}",
            flush=True,
        )

    overall = {
        "samples": len(all_solved),
        "solve_rate": float(np.mean(all_solved)) if all_solved else 0.0,
        "avg_len_solved": float(np.mean(all_len)) if all_len else None,
        "avg_ms": float(np.mean(all_ms)) if all_ms else None,
    }
    return {"overall": overall, "by_depth": by_depth}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Weighted A* evaluation of a DAVI value checkpoint.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--depths", nargs="+", default=("1", "2", "3"))
    parser.add_argument("--samples-per-depth", type=int, default=50)
    parser.add_argument("--weight", type=float, default=0.6)
    parser.add_argument("--policy-weight", type=float, default=0.0)
    parser.add_argument("--batch-expansion", type=int, default=100)
    parser.add_argument("--max-nodes", type=int, default=1_000_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    import torch

    device = torch.device(args.device)
    model, metadata = load_policy_value_checkpoint(args.model, map_location=device)
    model.to(device)
    model.eval()

    report = {
        "benchmark": "davi-weighted-astar",
        "model": {"path": str(args.model), "model_version": metadata.model_version},
        "config": {
            "depths": list(_parse_depths(args.depths)),
            "samples_per_depth": args.samples_per_depth,
            "weight": args.weight,
            "policy_weight": args.policy_weight,
            "batch_expansion": args.batch_expansion,
            "max_nodes": args.max_nodes,
            "seed": args.seed,
        },
        "summary": evaluate(
            model,
            device,
            depths=_parse_depths(args.depths),
            samples_per_depth=args.samples_per_depth,
            weight=args.weight,
            policy_weight=args.policy_weight,
            batch_expansion=args.batch_expansion,
            max_nodes=args.max_nodes,
            seed=args.seed,
        ),
    }

    if args.out is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
