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
from typing import Any, Sequence

import numpy as np

from rubic_rl import cube_ops as C
from rubic_rl.models.policy_value import load_policy_value_checkpoint
from rubic_rl.search import make_action_policy_fn, make_value_fn, weighted_astar_solve


def _parse_depths(values: Sequence[str]) -> tuple[int, ...]:
    raw: list[str] = []
    for value in values:
        raw.extend(part for part in value.split(",") if part)
    return tuple(int(depth) for depth in raw)


def _summarize_cases(
    cases: Sequence[dict[str, Any]],
    *,
    depths: tuple[int, ...],
) -> dict:
    by_depth: dict[str, dict] = {}
    all_solved: list[bool] = []
    all_len: list[int] = []
    all_ms: list[float] = []

    for depth in depths:
        depth_cases = [case for case in cases if int(case["depth"]) == depth]
        solved_flags = [bool(case["solved"]) for case in depth_cases]
        lengths = [int(case["length"]) for case in depth_cases if case["solved"]]
        expanded = [int(case["expanded"]) for case in depth_cases]
        durations = [float(case["ms"]) for case in depth_cases]
        by_depth[str(depth)] = {
            "samples": len(depth_cases),
            "solve_rate": float(np.mean(solved_flags)) if solved_flags else 0.0,
            "avg_len_solved": float(np.mean(lengths)) if lengths else None,
            "avg_expanded": float(np.mean(expanded)) if expanded else None,
            "avg_ms": float(np.mean(durations)) if durations else None,
        }
        all_solved.extend(solved_flags)
        all_len.extend(lengths)
        all_ms.extend(durations)

    overall = {
        "samples": len(all_solved),
        "solve_rate": float(np.mean(all_solved)) if all_solved else 0.0,
        "avg_len_solved": float(np.mean(all_len)) if all_len else None,
        "avg_ms": float(np.mean(all_ms)) if all_ms else None,
    }
    return {"overall": overall, "by_depth": by_depth}


def _write_report(path: Path, report: dict[str, Any], cases: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = dict(report)
    report["cases"] = list(cases)
    report["summary"] = _summarize_cases(
        cases,
        depths=tuple(int(depth) for depth in report["config"]["depths"]),
    )
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    report: dict[str, Any] | None = None,
    out: Path | None = None,
    resume_cases: Sequence[dict[str, Any]] | None = None,
    flush_every: int = 1,
) -> dict:
    value_fn = make_value_fn(model, device)
    action_policy_fn = make_action_policy_fn(model, device) if policy_weight > 0.0 else None
    rng = np.random.default_rng(seed)
    cases: list[dict[str, Any]] = list(resume_cases or [])
    completed = {(int(case["depth"]), int(case["index"])) for case in cases}
    since_flush = 0

    for depth in depths:
        if depth > 0:
            states, _ = C.scramble_batch(
                samples_per_depth, depth, rng, min_depth=depth
            )
        else:
            states = C.solved_batch(samples_per_depth)
        for index, row in enumerate(states):
            if (depth, index) in completed:
                continue
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
            case = {
                "depth": depth,
                "index": index,
                "solved": bool(result["solved"]),
                "length": int(result["length"]),
                "expanded": int(result["expanded"]),
                "generated": int(result["generated"]),
                "ms": (time.perf_counter() - start) * 1000.0,
            }
            cases.append(case)
            since_flush += 1
            print(
                f"[davi-eval] depth {depth:>2} case {index + 1:>3}/{samples_per_depth}: "
                f"solved={case['solved']} length={case['length']} "
                f"expanded={case['expanded']} ms={case['ms']:.0f}",
                flush=True,
            )
            if out is not None and report is not None and since_flush >= flush_every:
                _write_report(out, report, cases)
                since_flush = 0

        depth_summary = _summarize_cases(cases, depths=(depth,))["by_depth"][str(depth)]
        print(
            f"[davi-eval] depth {depth:>2}: solve_rate={depth_summary['solve_rate']:.3f} "
            f"avg_len={depth_summary['avg_len_solved']} "
            f"avg_ms={depth_summary['avg_ms']:.0f}",
            flush=True,
        )

    if out is not None and report is not None:
        _write_report(out, report, cases)
    return _summarize_cases(cases, depths=depths)


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
    parser.add_argument("--resume", action="store_true", help="Resume from an existing --out report and skip completed cases.")
    parser.add_argument("--flush-every", type=int, default=1, help="Write --out after this many completed cases.")
    args = parser.parse_args(argv)
    if args.resume and args.out is None:
        raise ValueError("--resume requires --out so completed cases can be loaded.")

    import torch

    device = torch.device(args.device)
    model, metadata = load_policy_value_checkpoint(args.model, map_location=device)
    model.to(device)
    model.eval()

    depths = _parse_depths(args.depths)
    config = {
        "depths": list(depths),
        "samples_per_depth": args.samples_per_depth,
        "weight": args.weight,
        "policy_weight": args.policy_weight,
        "batch_expansion": args.batch_expansion,
        "max_nodes": args.max_nodes,
        "seed": args.seed,
    }
    resume_cases: list[dict[str, Any]] = []
    if args.resume and args.out is not None and args.out.exists():
        existing = json.loads(args.out.read_text(encoding="utf-8"))
        existing_config = existing.get("config", {})
        if existing_config != config:
            raise ValueError(
                "--resume requires the existing report config to match the current command."
            )
        resume_cases = list(existing.get("cases", []))
        print(f"[davi-eval] resuming {len(resume_cases)} completed cases from {args.out}", flush=True)

    report = {
        "benchmark": "davi-weighted-astar",
        "model": {"path": str(args.model), "model_version": metadata.model_version},
        "config": config,
    }
    report["summary"] = evaluate(
        model,
        device,
        depths=depths,
        samples_per_depth=args.samples_per_depth,
        weight=args.weight,
        policy_weight=args.policy_weight,
        batch_expansion=args.batch_expansion,
        max_nodes=args.max_nodes,
        seed=args.seed,
        report=report,
        out=args.out,
        resume_cases=resume_cases,
        flush_every=max(1, args.flush_every),
    )

    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
