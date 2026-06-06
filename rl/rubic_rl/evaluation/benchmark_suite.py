from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Sequence

import numpy as np

from app.cube import Cube
from rubic_rl import cube_ops as C
from rubic_rl.evaluation.search_eval import (
    BenchmarkCase,
    BenchmarkEpisodeResult,
    SearchBenchmarkConfig,
    rank_strategy_reports,
    run_search_benchmark,
    summarize_benchmark_results,
)
from rubic_rl.models.policy_value import load_policy_value_checkpoint
from rubic_rl.policies import PolicyType, load_policy
from rubic_rl.search import make_action_policy_fn, make_value_fn, weighted_astar_solve


@dataclass(frozen=True)
class WeightedAStarBenchmarkConfig:
    weight: float = 0.6
    policy_weight: float = 0.0
    batch_expansion: int = 100
    max_nodes: int = 1_000_000

    def __post_init__(self) -> None:
        if self.batch_expansion <= 0:
            raise ValueError("batch_expansion must be positive")
        if self.max_nodes <= 0:
            raise ValueError("max_nodes must be positive")

    def to_json_dict(self) -> dict[str, float | int]:
        return {
            "weight": self.weight,
            "policy_weight": self.policy_weight,
            "batch_expansion": self.batch_expansion,
            "max_nodes": self.max_nodes,
        }


SUITE_PRESETS: dict[str, SearchBenchmarkConfig] = {
    "smoke": SearchBenchmarkConfig(
        depths=(1, 2, 3),
        samples_per_depth=10,
        max_depth=10,
        beam_width=5,
        top_k=5,
        seed=20260606,
    ),
    "shallow": SearchBenchmarkConfig(
        depths=(1, 2, 3, 4, 5),
        samples_per_depth=50,
        max_depth=20,
        beam_width=10,
        top_k=8,
        seed=20260606,
    ),
    "mid": SearchBenchmarkConfig(
        depths=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        samples_per_depth=50,
        max_depth=25,
        beam_width=10,
        top_k=8,
        seed=20260606,
    ),
    "search-heavy": SearchBenchmarkConfig(
        depths=(5, 6, 7, 8, 9, 10),
        samples_per_depth=50,
        max_depth=25,
        beam_width=20,
        top_k=10,
        seed=20260606,
    ),
}


def generate_suite_cases(config: SearchBenchmarkConfig) -> tuple[BenchmarkCase, ...]:
    rng = np.random.default_rng(config.seed)
    cases: list[BenchmarkCase] = []
    for depth in config.depths:
        for sample_index in range(config.samples_per_depth):
            case_seed = _next_seed(rng, config.seed)
            scramble = tuple(str(move) for move in _generate_scramble(depth, case_seed))
            cases.append(
                BenchmarkCase(
                    depth=depth,
                    sample_index=sample_index,
                    seed=case_seed,
                    scramble=scramble,
                )
            )
    return tuple(cases)


def write_benchmark_cases(
    path: str | Path,
    *,
    suite_name: str,
    config: SearchBenchmarkConfig,
    cases: Sequence[BenchmarkCase],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "suite": suite_name,
        "config": config.to_json_dict(),
        "cases": [case.to_json_dict() for case in cases],
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def load_benchmark_cases(path: str | Path) -> tuple[BenchmarkCase, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_cases = payload.get("cases", [])
    return tuple(
        BenchmarkCase(
            depth=int(case["depth"]),
            sample_index=int(case["sample_index"]),
            seed=None if case.get("seed") is None else int(case["seed"]),
            scramble=tuple(str(move) for move in case["scramble"]),
        )
        for case in raw_cases
    )


def run_benchmark_suite(
    *,
    suite_name: str,
    config: SearchBenchmarkConfig,
    cases: Sequence[BenchmarkCase],
    policy_models: Sequence[tuple[str, str | Path, PolicyType]] = (),
    davi_models: Sequence[tuple[str, str | Path]] = (),
    weighted_astar_config: WeightedAStarBenchmarkConfig | None = None,
    include_episodes: bool = False,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    weighted_astar_config = weighted_astar_config or WeightedAStarBenchmarkConfig()
    entrants: list[dict[str, Any]] = []

    for label, model_path, policy_type in policy_models:
        policy = load_policy(model_path, policy_type=policy_type)
        report = run_search_benchmark(
            policy,
            config,
            cases=cases,
            include_episodes=include_episodes,
            model_path=model_path,
            policy_type=policy_type,
        )
        for strategy in report["strategies"]:
            entrants.append(
                {
                    "label": f"{label}:{strategy['label']}",
                    "track": "policy",
                    "strategy": strategy["strategy"],
                    "summary": strategy["summary"],
                    "model": {
                        "path": str(model_path),
                        "policy_type": policy_type,
                    },
                    **({"episodes": strategy["episodes"]} if include_episodes else {}),
                }
            )

    for label, model_path in davi_models:
        entrants.append(
            run_davi_benchmark(
                label=label,
                model_path=model_path,
                cases=cases,
                config=weighted_astar_config,
                include_episodes=include_episodes,
            )
        )

    report: dict[str, Any] = {
        "benchmark": "rl-benchmark-suite",
        "suite": suite_name,
        "config": config.to_json_dict(),
        "weighted_astar": weighted_astar_config.to_json_dict(),
        "cases": [case.to_json_dict() for case in cases],
        "entrants": entrants,
        "ranking": rank_strategy_reports(entrants),
    }
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_davi_benchmark(
    *,
    label: str,
    model_path: str | Path,
    cases: Sequence[BenchmarkCase],
    config: WeightedAStarBenchmarkConfig,
    include_episodes: bool = False,
) -> dict[str, Any]:
    import torch

    device = torch.device("cpu")
    model, metadata = load_policy_value_checkpoint(model_path, map_location=device)
    model.to(device)
    model.eval()
    value_fn = make_value_fn(model, device)
    action_policy_fn = (
        make_action_policy_fn(model, device) if config.policy_weight > 0.0 else None
    )

    results: list[BenchmarkEpisodeResult] = []
    for case in cases:
        started_at = perf_counter()
        cube = Cube.solved().apply_sequence(case.scramble, record_history=False)
        state = C.cube_to_state(cube)
        solved = weighted_astar_solve(
            state,
            value_fn,
            action_policy_fn=action_policy_fn,
            weight=config.weight,
            policy_weight=config.policy_weight,
            batch_expansion=config.batch_expansion,
            max_nodes=config.max_nodes,
        )
        duration_ms = (perf_counter() - started_at) * 1000.0
        results.append(
            BenchmarkEpisodeResult(
                depth=case.depth,
                sample_index=case.sample_index,
                seed=case.seed,
                scramble=case.scramble,
                status="solved" if solved["solved"] else "failed",
                solved=bool(solved["solved"]),
                moves=tuple(str(move) for move in solved["moves"]),
                move_count=int(solved["length"]),
                duration_ms=duration_ms,
                expanded_states=int(solved["expanded"]),
                visited_states=int(solved.get("visited", solved["generated"])),
                depth_reached=int(solved.get("depth_reached", solved["length"])),
            )
        )

    report: dict[str, Any] = {
        "label": label,
        "track": "davi",
        "strategy": "weighted-astar",
        "summary": summarize_benchmark_results(results),
        "model": {
            "path": str(model_path),
            "model_version": metadata.model_version,
            "value_scale": metadata.value_scale,
        },
    }
    if include_episodes:
        report["episodes"] = [result.to_json_dict() for result in results]
    return report


def preset_config(name: str) -> SearchBenchmarkConfig:
    try:
        return SUITE_PRESETS[name]
    except KeyError as error:
        raise ValueError(f"Unknown suite preset: {name!r}") from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a shared RL benchmark suite.")
    parser.add_argument(
        "--suite",
        choices=tuple(sorted(SUITE_PRESETS)),
        default="shallow",
        help="Named benchmark suite preset.",
    )
    parser.add_argument(
        "--cases-out",
        type=Path,
        help="Optional JSON path to persist the benchmark cases for reuse.",
    )
    parser.add_argument(
        "--cases-in",
        type=Path,
        help="Optional JSON case file previously written by --cases-out.",
    )
    parser.add_argument(
        "--policy-model",
        action="append",
        default=[],
        help="Policy entrant in label=path format. May be passed multiple times.",
    )
    parser.add_argument(
        "--policy-type",
        choices=("auto", "linear", "mlp", "torch"),
        default="auto",
        help="Checkpoint policy type to use for every --policy-model entrant.",
    )
    parser.add_argument(
        "--davi-model",
        action="append",
        default=[],
        help="DAVI entrant in label=path format. May be passed multiple times.",
    )
    parser.add_argument("--include-episodes", action="store_true")
    parser.add_argument("--out", type=Path, help="Optional JSON report output path.")
    parser.add_argument("--weight", type=float, default=0.6)
    parser.add_argument("--policy-weight", type=float, default=0.0)
    parser.add_argument("--batch-expansion", type=int, default=100)
    parser.add_argument("--max-nodes", type=int, default=1_000_000)
    args = parser.parse_args(argv)

    config = preset_config(args.suite)
    cases = (
        load_benchmark_cases(args.cases_in)
        if args.cases_in is not None
        else generate_suite_cases(config)
    )
    if args.cases_out is not None:
        write_benchmark_cases(args.cases_out, suite_name=args.suite, config=config, cases=cases)

    report = run_benchmark_suite(
        suite_name=args.suite,
        config=config,
        cases=cases,
        policy_models=[
            (*_parse_labelled_path(spec), args.policy_type) for spec in args.policy_model
        ],
        davi_models=[_parse_labelled_path(spec) for spec in args.davi_model],
        weighted_astar_config=WeightedAStarBenchmarkConfig(
            weight=args.weight,
            policy_weight=args.policy_weight,
            batch_expansion=args.batch_expansion,
            max_nodes=args.max_nodes,
        ),
        include_episodes=args.include_episodes,
        output_path=args.out,
    )
    if args.out is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _parse_labelled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("Expected label=path format")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    raw_path = raw_path.strip()
    if not label or not raw_path:
        raise ValueError("Expected non-empty label=path format")
    return label, Path(raw_path)


def _generate_scramble(depth: int, seed: int | None) -> tuple[str, ...]:
    from app.cube import generate_scramble

    return generate_scramble(depth=depth, seed=seed)


def _next_seed(rng: np.random.Generator, root_seed: int | None) -> int | None:
    if root_seed is None:
        return None
    return int(rng.integers(0, np.iinfo(np.int32).max))


__all__ = [
    "SUITE_PRESETS",
    "WeightedAStarBenchmarkConfig",
    "generate_suite_cases",
    "load_benchmark_cases",
    "preset_config",
    "run_benchmark_suite",
    "run_davi_benchmark",
    "write_benchmark_cases",
]
