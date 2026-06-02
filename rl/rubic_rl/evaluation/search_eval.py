from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import numpy as np

from app.cube import Cube, generate_scramble, parse_moves
from app.solvers.rl_search import (
    RL_ACTION_MOVES,
    PolicyGuidedSearch,
    SearchConfig,
    predict_top_action,
)
from rubic_rl.policies import PolicyType, load_policy


class SearchPolicy(Protocol):
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return integer action IDs for a batch of feature vectors."""


Clock = Callable[[], float]


@dataclass(frozen=True)
class SearchBenchmarkConfig:
    depths: tuple[int, ...] = (1, 2, 3)
    samples_per_depth: int = 100
    max_depth: int = 30
    beam_width: int = 5
    top_k: int = 5
    seed: int | None = 0

    def __post_init__(self) -> None:
        depths = tuple(int(depth) for depth in self.depths)
        if not depths:
            raise ValueError("depths must contain at least one value")
        if any(depth < 0 for depth in depths):
            raise ValueError("depths cannot contain negative values")
        if self.samples_per_depth <= 0:
            raise ValueError("samples_per_depth must be positive")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if self.beam_width <= 0:
            raise ValueError("beam_width must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")

        object.__setattr__(self, "depths", depths)

    def to_json_dict(self) -> dict[str, int | list[int] | None]:
        return {
            "depths": list(self.depths),
            "samples_per_depth": self.samples_per_depth,
            "max_depth": self.max_depth,
            "beam_width": self.beam_width,
            "top_k": self.top_k,
            "seed": self.seed,
        }

    def search_config(self) -> SearchConfig:
        return SearchConfig(
            max_depth=self.max_depth,
            beam_width=self.beam_width,
            top_k=self.top_k,
        )


@dataclass(frozen=True)
class BenchmarkCase:
    depth: int
    sample_index: int
    seed: int | None
    scramble: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.depth < 0:
            raise ValueError("depth cannot be negative")
        if self.sample_index < 0:
            raise ValueError("sample_index cannot be negative")
        object.__setattr__(self, "scramble", parse_moves(self.scramble))

    def to_json_dict(self) -> dict[str, int | None | list[str]]:
        return {
            "depth": self.depth,
            "sample_index": self.sample_index,
            "seed": self.seed,
            "scramble": list(self.scramble),
        }


@dataclass(frozen=True)
class BenchmarkEpisodeResult:
    depth: int
    sample_index: int
    seed: int | None
    scramble: tuple[str, ...]
    status: str
    solved: bool
    moves: tuple[str, ...]
    move_count: int
    duration_ms: float
    expanded_states: int
    visited_states: int
    depth_reached: int

    def to_json_dict(self) -> dict[str, int | float | bool | None | list[str]]:
        return {
            "depth": self.depth,
            "sample_index": self.sample_index,
            "seed": self.seed,
            "scramble": list(self.scramble),
            "status": self.status,
            "solved": self.solved,
            "moves": list(self.moves),
            "move_count": self.move_count,
            "duration_ms": self.duration_ms,
            "expanded_states": self.expanded_states,
            "visited_states": self.visited_states,
            "depth_reached": self.depth_reached,
        }


@dataclass(frozen=True)
class BenchmarkSummary:
    attempts: int
    solved: int
    solve_rate: float
    avg_move_count_solved: float | None
    avg_move_count_all: float | None
    avg_duration_ms: float | None
    avg_expanded_states: float | None
    avg_visited_states: float | None
    avg_depth_reached: float | None

    def to_json_dict(self) -> dict[str, int | float | None]:
        return {
            "attempts": self.attempts,
            "solved": self.solved,
            "solve_rate": self.solve_rate,
            "avg_move_count_solved": self.avg_move_count_solved,
            "avg_move_count_all": self.avg_move_count_all,
            "avg_duration_ms": self.avg_duration_ms,
            "avg_expanded_states": self.avg_expanded_states,
            "avg_visited_states": self.avg_visited_states,
            "avg_depth_reached": self.avg_depth_reached,
        }


def generate_benchmark_cases(config: SearchBenchmarkConfig) -> tuple[BenchmarkCase, ...]:
    rng = np.random.default_rng(config.seed)
    cases: list[BenchmarkCase] = []
    for depth in config.depths:
        for sample_index in range(config.samples_per_depth):
            case_seed = _next_seed(rng, config.seed)
            cases.append(
                BenchmarkCase(
                    depth=depth,
                    sample_index=sample_index,
                    seed=case_seed,
                    scramble=generate_scramble(depth=depth, seed=case_seed),
                )
            )
    return tuple(cases)


def evaluate_greedy_on_cases(
    policy: SearchPolicy,
    cases: Iterable[BenchmarkCase],
    *,
    max_depth: int,
    timer: Clock = perf_counter,
) -> tuple[BenchmarkEpisodeResult, ...]:
    if max_depth <= 0:
        raise ValueError("max_depth must be positive")

    return tuple(_evaluate_greedy_case(policy, case, max_depth, timer) for case in cases)


def evaluate_search_on_cases(
    policy: SearchPolicy,
    cases: Iterable[BenchmarkCase],
    *,
    search_config: SearchConfig,
    timer: Clock = perf_counter,
) -> tuple[BenchmarkEpisodeResult, ...]:
    searcher = PolicyGuidedSearch(policy, config=search_config)
    return tuple(_evaluate_search_case(searcher, case, timer) for case in cases)


def summarize_benchmark_results(
    results: Sequence[BenchmarkEpisodeResult],
) -> dict[str, dict[str, int | float | None] | dict[str, dict[str, int | float | None]]]:
    by_depth: dict[str, dict[str, int | float | None]] = {}
    for depth in sorted({result.depth for result in results}):
        depth_results = [result for result in results if result.depth == depth]
        by_depth[str(depth)] = _summarize_group(depth_results).to_json_dict()

    return {
        "overall": _summarize_group(results).to_json_dict(),
        "by_depth": by_depth,
    }


def run_search_benchmark(
    policy: SearchPolicy,
    config: SearchBenchmarkConfig | None = None,
    *,
    cases: Iterable[BenchmarkCase] | None = None,
    include_episodes: bool = False,
    model_path: str | Path | None = None,
    policy_type: PolicyType = "auto",
    output_path: str | Path | None = None,
    timer: Clock = perf_counter,
) -> dict[str, Any]:
    config = config or SearchBenchmarkConfig()
    benchmark_cases = tuple(cases) if cases is not None else generate_benchmark_cases(config)

    greedy_results = evaluate_greedy_on_cases(
        policy,
        benchmark_cases,
        max_depth=config.max_depth,
        timer=timer,
    )
    search_results = evaluate_search_on_cases(
        policy,
        benchmark_cases,
        search_config=config.search_config(),
        timer=timer,
    )
    strategies = [
        _strategy_report(
            "greedy",
            "greedy-policy-rollout",
            greedy_results,
            include_episodes=include_episodes,
        ),
        _strategy_report(
            "beam_search",
            "policy-guided-beam-search",
            search_results,
            include_episodes=include_episodes,
        ),
    ]

    report: dict[str, Any] = {
        "benchmark": "rl-search",
        "config": config.to_json_dict(),
        "cases": [case.to_json_dict() for case in benchmark_cases],
        "strategies": strategies,
        "ranking": rank_strategy_reports(strategies),
    }
    if model_path is not None:
        report["model"] = {
            "path": str(model_path),
            "policy_type": policy_type,
        }

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return report


def rank_strategy_reports(strategy_reports: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        strategy_reports,
        key=lambda report: (
            -_overall_float(report, "solve_rate", default=0.0),
            _overall_float(report, "avg_move_count_solved", default=float("inf")),
            _overall_float(report, "avg_duration_ms", default=float("inf")),
            _overall_float(report, "avg_expanded_states", default=float("inf")),
            str(report["label"]),
        ),
    )

    ranking: list[dict[str, Any]] = []
    for rank, report in enumerate(ranked, start=1):
        overall = report["summary"]["overall"]
        ranking.append(
            {
                "rank": rank,
                "label": report["label"],
                "attempts": overall["attempts"],
                "solved": overall["solved"],
                "solve_rate": overall["solve_rate"],
                "avg_move_count_solved": overall["avg_move_count_solved"],
                "avg_duration_ms": overall["avg_duration_ms"],
                "avg_expanded_states": overall["avg_expanded_states"],
            }
        )
    return ranking


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark greedy and beam-search Rubik policy inference."
    )
    parser.add_argument("--model", required=True, type=Path, help="Input model checkpoint.")
    parser.add_argument(
        "--policy-type",
        choices=("auto", "linear", "mlp", "torch"),
        default="auto",
        help="Checkpoint policy type. 'auto' detects .npz and .pt checkpoints.",
    )
    parser.add_argument(
        "--depths",
        nargs="+",
        default=("1", "2", "3"),
        help="Scramble depths, for example: --depths 1 2 3 or --depths 1,2,3.",
    )
    parser.add_argument("--samples-per-depth", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=30)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--seed",
        default="0",
        help="Benchmark seed. Use 'none' for non-deterministic scrambles.",
    )
    parser.add_argument(
        "--include-episodes",
        action="store_true",
        help="Include per-case results in the JSON output.",
    )
    parser.add_argument("--out", type=Path, help="Optional JSON report output path.")
    args = parser.parse_args(argv)

    policy = load_policy(args.model, policy_type=args.policy_type)
    config = SearchBenchmarkConfig(
        depths=_parse_depth_args(args.depths),
        samples_per_depth=args.samples_per_depth,
        max_depth=args.max_depth,
        beam_width=args.beam_width,
        top_k=args.top_k,
        seed=_parse_seed(args.seed),
    )
    report = run_search_benchmark(
        policy,
        config,
        include_episodes=args.include_episodes,
        model_path=args.model,
        policy_type=args.policy_type,
        output_path=args.out,
    )
    if args.out is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _evaluate_greedy_case(
    policy: SearchPolicy,
    case: BenchmarkCase,
    max_depth: int,
    timer: Clock,
) -> BenchmarkEpisodeResult:
    started_at = timer()
    cube = Cube.solved().apply_sequence(case.scramble, record_history=False)
    moves: list[str] = []
    visited = {cube.stickers}
    expanded_states = 0
    status = "solved" if cube.is_solved() else "failed"

    while not cube.is_solved() and len(moves) < max_depth:
        expanded_states += 1
        action = predict_top_action(policy, cube)
        move = RL_ACTION_MOVES[action]
        cube = cube.apply_move(move, record_history=False)
        visited.add(cube.stickers)
        moves.append(move)
        if cube.is_solved():
            status = "solved"
            break

    duration_ms = (timer() - started_at) * 1000.0
    solved = cube.is_solved()
    return BenchmarkEpisodeResult(
        depth=case.depth,
        sample_index=case.sample_index,
        seed=case.seed,
        scramble=case.scramble,
        status="solved" if solved else status,
        solved=solved,
        moves=tuple(moves),
        move_count=len(moves),
        duration_ms=duration_ms,
        expanded_states=expanded_states,
        visited_states=len(visited),
        depth_reached=len(moves),
    )


def _evaluate_search_case(
    searcher: PolicyGuidedSearch,
    case: BenchmarkCase,
    timer: Clock,
) -> BenchmarkEpisodeResult:
    started_at = timer()
    cube = Cube.solved().apply_sequence(case.scramble, record_history=False)
    result = searcher.solve(cube)
    duration_ms = (timer() - started_at) * 1000.0
    return BenchmarkEpisodeResult(
        depth=case.depth,
        sample_index=case.sample_index,
        seed=case.seed,
        scramble=case.scramble,
        status=result.status,
        solved=result.status == "solved",
        moves=result.moves,
        move_count=len(result.moves),
        duration_ms=duration_ms,
        expanded_states=result.expanded_states,
        visited_states=result.visited_states,
        depth_reached=result.depth_reached,
    )


def _strategy_report(
    label: str,
    strategy: str,
    results: Sequence[BenchmarkEpisodeResult],
    *,
    include_episodes: bool,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "label": label,
        "strategy": strategy,
        "summary": summarize_benchmark_results(results),
    }
    if include_episodes:
        report["episodes"] = [result.to_json_dict() for result in results]
    return report


def _summarize_group(results: Sequence[BenchmarkEpisodeResult]) -> BenchmarkSummary:
    attempts = len(results)
    solved_results = [result for result in results if result.solved]
    solved = len(solved_results)
    return BenchmarkSummary(
        attempts=attempts,
        solved=solved,
        solve_rate=0.0 if attempts == 0 else solved / attempts,
        avg_move_count_solved=_mean_or_none(
            result.move_count for result in solved_results
        ),
        avg_move_count_all=_mean_or_none(result.move_count for result in results),
        avg_duration_ms=_mean_or_none(result.duration_ms for result in results),
        avg_expanded_states=_mean_or_none(result.expanded_states for result in results),
        avg_visited_states=_mean_or_none(result.visited_states for result in results),
        avg_depth_reached=_mean_or_none(result.depth_reached for result in results),
    )


def _mean_or_none(values: Iterable[int | float]) -> float | None:
    collected = [float(value) for value in values]
    if not collected:
        return None
    return sum(collected) / len(collected)


def _next_seed(rng: np.random.Generator, root_seed: int | None) -> int | None:
    if root_seed is None:
        return None
    return int(rng.integers(0, np.iinfo(np.int32).max))


def _overall_float(report: dict[str, Any], field: str, *, default: float) -> float:
    value = report["summary"]["overall"][field]
    if value is None:
        return default
    return float(value)


def _parse_depth_args(values: Sequence[str]) -> tuple[int, ...]:
    raw_depths: list[str] = []
    for value in values:
        raw_depths.extend(part for part in value.split(",") if part)
    return tuple(int(depth) for depth in raw_depths)


def _parse_seed(value: str | None) -> int | None:
    if value is None or value.lower() == "none":
        return None
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
